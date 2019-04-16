import math
import os
import random
import logging
from tqdm import tqdm

import muon.project.panoptes as panoptes
import muon.config
from muon.subjects.subjects import Subjects
from muon.database.utils import StorageObject, StorageAttribute, \
        StoredAttribute
from muon.images.image import Image
from muon.images.single_image import SingleImage

logger = logging.getLogger(__name__)


class ImageGroupParent(StorageObject):

    TYPES = {s: i for i, s in enumerate(['grid', 'single'])}
    image_count = StorageAttribute('image_count')

    def __init__(self, group_id, database, attrs=None, online=False):
    # def __init__(self, group_id, database, attrs=None):
        """
        Parameters:
            group
            cluster_name
            images
            image_size
            width
            description
            permutations
        """

        super().__init__(database, online)
        self.group_id = group_id

        if attrs is None:
            with self.conn as conn:
                attrs = database.ImageGroup.get_group(conn, group_id)
        self.group_type = attrs['group_type']
        self.image_size = attrs['image_size']
        self.image_width = attrs['image_width']
        self.description = attrs['description']
        self.permutations = attrs['permutations']
        self.cluster_name = attrs['cluster_name']
        
        self.storage = {s.name: s for s in [
            StoredAttribute('image_count', attrs['image_count'])]}

        self.images = ImageLoader(group_id, self.image_count, database,
                                  self.group_type, online)

        # self.cluster_name = cluster_name




        # self.image_size = kwargs.get('image_size', 36)
        # self.image_width = kwargs.get('image_width', 6)
        # self.description = kwargs.get('description', None)
        # self.permutations = kwargs.get('permutations', 1)

        # self.images = images or {}
        # self.zoo_map = None

    @classmethod
    def new(cls, database, group_id, image_size,
            group_type, image_width, description,
            permutations, cluster_name):

        if group_id is None:
            with database.conn as conn:
                group_id = database.ImageGroup.next_id(conn)

        attrs = {
            'image_size': image_size,
            'group_type': group_type,
            'image_width': image_width,
            'description': description,
            'permutations': permutations,
            'cluster_name': cluster_name,
            'image_count': 0
        }

        group = cls(group_id, database, attrs=attrs)

        with database.conn as conn:
            database.ImageGroup.add_group(conn, group)
            conn.commit()

        return group

    def save(self):
        updates = {}
        for k, v in self.storage.items():
            if v.has_changed:
                updates[k] = v.value
                v.has_changed = False

                if k == 'image_count':
                    self.images.image_count = v.value

        if updates:
            with self.conn as conn:
                self.database.ImageGroup \
                    .update_group(conn, self.group_id, updates)
                conn.commit()

    def _create_images(self, kwargs):
        pass

    def metadata(self):
        return {
            'cluster_name': self.cluster_name,
            'image_size': self.image_size,
            'image_width': self.image_width,
            'group_id': self.group_id,
            'description': self.description,
            'permutations': self.permutations
        }

    def __str__(self):
        s = 'group {} cluster_name {} images {} metadata {}'.format(
            self.group_id, self.cluster_name,
            self.image_count, self.metadata())
        return s

    def __repr__(self):
        return str(self)

    def upload_subjects(self, path, existing_subjects=None):
        """
        Upload generated images to Panoptes
        """
        project_id = muon.config.Panoptes().project_id
        uploader = panoptes.Uploader(project_id, self.group_id)

        if existing_subjects:
            images = self.images
        else:
            images = self.images.upload_iter()

        print('Creating Panoptes subjects')
        for image in tqdm(images):

            if existing_subjects:
                if image.image_id in existing_subjects:
                    zoo_id = existing_subjects[image.image_id]
                    if image.zoo_id != zoo_id:
                        logger.debug('Updating image zoo id')
                        image.zoo_id = zoo_id
                    else:
                        print('Skipping {}'.format(image))
                        continue

                elif image.zoo_id is not None:
                    print(image)
                    raise Exception('Found subject with existing zoo_id '
                                    'but cannot find associated subject in export')
            elif image.zoo_id is not None:
                print('skipping {}'.format(image))
                continue
            elif image.zoo_id is None:
                image.zoo_id = -1
                fname = os.path.join(
                    path, 'group_%d' % self.group_id, image.fname())

                subject = panoptes.Subject()
                subject.add_location(fname)
                subject.metadata.update(image.dump_manifest())

                subject = uploader.add_subject(subject)
                image.zoo_id = subject.id

                logger.info(image)
            else:
                raise Exception

    def generate_images(self, subject_storage, dpi=None, path=None):
        """
        Generate subject images to be uploaded to Panoptes
        """
        path = os.path.join(path, 'group_%d' % self.group_id)
        if not os.path.isdir(path):
            os.mkdir(path)
        for image in tqdm(self.images.gen_iter(path)):
            if image.plot(self.image_width, subject_storage,
                          dpi=dpi, path=path):
                logger.info(image)


class ImageLoader:

    def __init__(self, group_id, image_count, database, group_type, online):
        self.database = database
        self.online = online

        self.group_id = group_id
        self.image_count = image_count
        self._loaded_images = []
        self._images = {}
        self._group_type = group_type 

        self.load_limit = 10000

    def __getitem__(self, image_id):
        if image_id not in self._images:
            image = self._load_image(image_id)
            if image is None:
                raise KeyError('No image {} in group {}'.format(
                    image_id, self.group_id))

            self._images[image_id] = image
            self._loaded_images.append(image_id)

            if self.load_limit > 0 and len(self._images) > self.load_limit:
                del self._images[self._loaded_images.pop(0)]
        return self._images[image_id]

    def load_all(self):
        with self.database.conn as conn:
            for image in tqdm(self.database.Image
                    .get_group_images(conn, self.group_id)):
                image = self._create_image(image)
                self._images[image.image_id] = image

    def gen_iter(self, path):
        with self.database.conn as conn:
            image_ids = list(self.database.Image
                .get_group_image_ids(conn, self.group_id, shuffle=True))

            for image_id in image_ids:
                image = self.database.Image.get_image(conn, image_id)
                image = self._create_image(image)
                if not os.path.isfile(os.path.join(path, image.fname())):
                    yield image

    def upload_iter(self):
        with self.database.conn as conn:
            image_ids = list(self.database.Image
                .get_group_image_ids(
                    conn, self.group_id, exclude_zoo=True, shuffle=True))

            for image_id in image_ids:
                image = self.database.Image.get_image(conn, image_id)
                yield self._create_image(image)

    def __iter__(self):
        with self.database.conn as conn:
            for image in self.database.Image  \
                    .get_group_images(conn, self.group_id, shuffle=True):

                yield self._create_image(image)

    def __len__(self):
        return self.image_count

    def _image_type(self):
        if self._group_type == ImageGroupParent.TYPES['grid']:
            return Image
        elif self._group_type == ImageGroupParent.TYPES['single']:
            return SingleImage

    def _load_image(self, image_id):
        Image_ = self._image_type()
        return Image_(image_id, self.database, online=self.online)

    def _create_image(self, image_dict):
        Image_ = self._image_type()
        return Image_(image_dict['image_id'], self.database,
                      online=self.online, attrs=image_dict)
