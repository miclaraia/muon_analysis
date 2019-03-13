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

logger = logging.getLogger(__name__)


class ImageGroup(StorageObject):

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
        self.image_size = attrs['image_size']
        self.image_width = attrs['image_width']
        self.description = attrs['description']
        self.permutations = attrs['permutations']
        self.cluster_name = attrs['cluster_name']
        
        self.storage = {s.name: s for s in [
            StoredAttribute('image_count', attrs['image_count'])]}

        self.images = ImageLoader(group_id, self.image_count, database, online)

        # self.cluster_name = cluster_name




        # self.image_size = kwargs.get('image_size', 36)
        # self.image_width = kwargs.get('image_width', 6)
        # self.description = kwargs.get('description', None)
        # self.permutations = kwargs.get('permutations', 1)

        # self.images = images or {}
        # self.zoo_map = None

    @classmethod
    def new(cls, database, cluster_name, cluster_assignments,
            group_id=None, **kwargs):
    # def new(cls, group_id, next_id,
            # subject_storage, cluster_name, cluster_assignments, **kwargs):
        """
        Parameters:
            next_id: callback function to get next image id
            cluster_assignments: {cluster: [subject_id]}
        """

        if group_id is None:
            with database.conn as conn:
                group_id = database.ImageGroup.next_id(conn)

        attrs = {
            'image_size': kwargs.get('image_size', 36),
            'image_width': kwargs.get('image_width', 6),
            'description': kwargs.get('description', None),
            'permutations': kwargs.get('permutations', 1),
            'cluster_name': cluster_name,
            'image_count': 0
        }

        group = cls(group_id, database, attrs=attrs)

        with database.conn as conn:
            database.ImageGroup.add_group(conn, group)
            conn.commit()

        group.create_images(cluster_assignments)

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

    def create_images(self, cluster_assignments):
        """
        Parameters:
            next_id: next id for a new image
            subjects
            cluster_assignments: {cluster: [subject_id]}
        """

        with self.conn as conn:
            _next_id = self.database.Image.next_id(conn)

        def f_next_id():
            i = 0
            while 1:
                yield i + _next_id
                i += 1
        next_id_iter = f_next_id()

        def next_id():
            return next(next_id_iter)

        for cluster in cluster_assignments:
            cluster_subjects = cluster_assignments[cluster]
            attrs = {
                'group_id': self.group_id,
                'cluster': cluster,
                'metadata': {},
                'commit': False,
            }

            with self.database.conn as conn:
                count = 0
                for image_subjects in self.split_subjects(cluster_subjects):
                    image = Image.new(self.database,
                                      image_id=next_id(),
                                      subjects=image_subjects, **attrs)
                    self.database.Image.add_image(conn, image)
                    count += 1

                self.image_count += count
                conn.commit()
        self.save()

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

    # def get_zoo(self, zoo_id):
        # if self.zoo_map is None:
            # zoo_map = {}
            # for i in self.iter():
                # if i.zoo_id:
                    # zoo_map[i.zoo_id] = i.id
            # self.zoo_map = zoo_map
        # return self.images[self.zoo_map[zoo_id]]

    # def list(self):
        # return list(self.iter())

    def split_subjects(self, subject_ids):
        """
        Subdivide a list of subjects into image groups, each of size
        determined in constructor call.
        """
        images = []
        subject_ids = subject_ids.copy()

        for _ in range(self.permutations):
            random.shuffle(subject_ids)

            # Sometimes the number of subjects in a cluster cannot be evenly
            # split into N even sized images. In this case we add enough
            # duplicate subjects to the last image to make it the same size
            if len(subject_ids) % self.image_size > 0:
                l = len(subject_ids)
                logger.debug('{}/36={:.1f}+{}'.format(
                    l, l/self.image_size, l%self.image_size))
                diff = self.image_size - (len(subject_ids) % self.image_size)
                logger.debug('adding {}'.format(diff))

                subject_ids += random.sample(subject_ids, diff)

            length = len(subject_ids)
            logger.debug('{} subjects, {} images'.format(
                length, length/self.image_size))
            w = math.ceil(length/self.image_size)
            for n in range(w):
                a = n*self.image_size
                b = min(length, a+self.image_size)
                subset = subject_ids[a:b]

                images.append(subset)
        return images

    # def remove_images(self, images):
        # for image in self.iter():
            # if image.id in images:
                # image.metadata['deleted'] = True

    def upload_subjects(self, path, existing_subjects=None):
        """
        Upload generated images to Panoptes
        """
        uploader = panoptes.Uploader(muon.config.project, self.group_id)

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

    def __init__(self, group_id, image_count, database, online):
        self.database = database
        self.online=online

        self.group_id = group_id
        self.image_count = image_count
        self._loaded_images = []
        self._images = {}

    def __getitem__(self, image_id):
        if image_id not in self._images:
            image = self._load_image(image_id)
            if image is None:
                raise KeyError('No image {} in group {}'.format(
                    image_id, self.group_id))

            self._images[image_id] = image
            self._loaded_images.append(image_id)

            if len(self._images) > 1000:
                del self._images[self._loaded_images.pop(0)]
        return self._images[image_id]

    def load_all(self):
        with self.database.conn as conn:
            image_ids = self.database.Image \
                .get_group_images(conn, self.group_id)

            for image_id in tqdm(image_ids):
                self._images[image_id] = self._load_image(image_id)

    def gen_iter(self, path):
        with self.database.conn as conn:
            image_ids = list(self.database.Image
                .get_group_images(conn, self.group_id))
            random.shuffle(image_ids)

            for image_id in image_ids:
                fname = Image._fname_static(image_id, self.group_id)
                if not os.path.isfile(os.path.join(path, fname)):
                    yield self._load_image(image_id)

    def upload_iter(self):
        with self.database.conn as conn:
            image_ids = list(self.database.Image
                .get_group_images(conn, self.group_id, ignore_zoo=True))
            random.shuffle(image_ids)

            for image_id in image_ids:
                yield self._load_image(image_id)

    def __iter__(self):
        with self.database.conn as conn:
            image_ids = list(self.database.Image
                .get_group_images(conn, self.group_id))
            image_ids = list(set(image_ids) - set(self._images.keys()))

            random.shuffle(image_ids)

            for image_id in image_ids:
                if image_id in self._images:
                    yield self._images[image_id]
                else:
                    yield self._load_image(image_id)

    def __len__(self):
        return self.image_count

    def _load_image(self, image_id):
        image = Image(image_id, self.database, online=self.online)
        return image
