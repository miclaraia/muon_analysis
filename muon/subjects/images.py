import math
import os
import json
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import logging
from tqdm import tqdm

import muon.project.panoptes as panoptes
import muon.config
from muon.subjects.subjects import Subjects

import muon.data

logger = logging.getLogger(__name__)


class Image:
    """
    A group of subjects which are uploaded to Panoptes as a single Image
    """

    def __init__(self, image_id, group_id,
                 cluster, subjects, metadata, zoo_id=None, image_meta=None):
        self.image_id = image_id
        self.group_id = group_id
        self.cluster = cluster
        self.subjects = subjects
        self.metadata = metadata
        self.zoo_id = zoo_id

        self.image_meta = image_meta
        
    def __str__(self):
        return 'id {} group {} cluster {} subjects {} ' \
               'metadata {} zooid {} image_meta {}'.format(
            self.image_id, self.group_id, self.cluster, len(self.subjects),
            str(self.metadata), self.zoo_id, self.image_meta)

    def __repr__(self):
        return str(self)

    def dump_manifest(self):
        data = OrderedDict([
            ('id', self.image_id),
            ('#group', self.group_id),
            ('image', self.fname()),
            ('#subjects', self.subjects)])
        hide = ['cluster']
        for k, v in self.metadata.items():
            if k not in data:
                if k in hide:
                    k = '#' + k
                data[k] = v

        return data

    def fname(self):
        """
        Filename to use for this subject group
        """
        return 'muon_group_%d_id_%d.png' % (self.group_id, self.image_id)

    def plot(self, width, subject_storage, path=None):
        """
        Generate and save a plot of this image
        """
        # subjects = subject_storage.get_subjects(self.subjects)
        subjects = {}
        for s in self.subjects:
            subject = subject_storage.get_subject(s)
            if s in subjects:
                subjects['{}-duplicate'.format(s)] = subject
            else:
                subjects[s] = subject

        fname = self.fname()

        # Skip if image already exists
        if fname in os.listdir(path):
            return

        if path:
            fname = os.path.join(path, fname)

        offset = .5
        dpi = 100
        fig, meta = Subjects(subjects).plot_subjects(
            w=width, grid=True, grid_args={'offset': offset}, meta=True)


        self.image_meta = Image.ImageMeta(dpi=dpi, offset=offset, **meta)
        # TODO make sure to actually update this data in the db
        # Probably need to add new columns

        fig.savefig(fname, dpi=dpi)

        plt.close(fig)

    def at_location(self, x, y):
        """
        Return the subject that should be at the given x,y coordinates
        """
        dpi = self.image_meta.dpi
        offset = self.image_meta.offset*dpi
        height = self.image_meta.height*dpi - offset
        width = self.image_meta.width*dpi - offset

        rows = self.image_meta.rows
        cols = self.image_meta.cols

        logger.debug(self.metadata)
        logger.debug(self.image_meta)

        if 'beta_image' in self.metadata:
            # This image was generated before the offset bug was discovered
            # and need to correct the vertical offset to get the right
            # boundary calculations
            height = self.image_meta.height*dpi*0.97 - offset
            width = self.image_meta.width*dpi*0.97 - offset
            y = y - 0.03*self.image_meta.height*dpi + offset

        y_ = height/rows
        x_ = width/cols
        logger.debug('x: {}'.format(x))
        logger.debug('y: {}'.format(y))
        logger.debug('offset: {}'.format(offset))
        logger.debug('width: {}'.format(width))
        logger.debug('height: {}'.format(height))

        logger.debug('x_: {}'.format(x_))
        logger.debug('y_: {}'.format(y_))

        x = (x-offset)//x_
        y = (y-offset)//y_

        i = int(x+cols*y)
        logger.debug(i)
        return self.subjects[i]

    class ImageMeta:
        def __init__(self,
                     dpi=None,
                     offset=None,
                     height=None,
                     width=None,
                     rows=None,
                     cols=None):
            self.dpi = dpi
            self.offset = offset
            self.height = height
            self.width = width
            self.rows = rows
            self.cols = cols

        def __str__(self):
            return str(self.__dict__)


class ImageGroup:

    def __init__(self, group_id, images, **kwargs):
        """
        
        Parameters:
            group
            images
            image_size
            width
            description
            permutations
        """

        self.group_id = group_id
        self.image_size = kwargs.get('image_size', 36)
        self.image_width = kwargs.get('image_width', 6)
        self.description = kwargs.get('description', None)
        self.permutations = kwargs.get('permutations', 1)

        self.images = images or {}
        self.zoo_map = None

    @classmethod
    def new(cls, group_id, next_id,
            subject_storage, cluster_assignments, **kwargs):
        """
        Parameters:
            next_id: callback function to get next image id
            cluster_assignments: {cluster: [subject_id]}
        """
        self = cls(group_id, None, **kwargs)
        images = self.create_images(
            group_id, next_id, subject_storage, cluster_assignments)
        self.images = images
        return self

        # images = {}
        # i = next_id
        # for c in cluster_assignments:
            # cluster_subjects = subjects.subset(cluster_assignments[c])
            # split_cluster = self.split_subjects(cluster_subjects)
            
            # for image_subjects in split_cluster:
                # meta = {'cluster': c}
                # images[i] = Image(i, self.group, image_subjects, meta)
                # i += 1

        # self.images = images
        # return i, self

    def create_images(
            self, group_id, next_id, subject_storage, cluster_assignments):
        """
        Parameters:
            next_id: next id for a new image
            subjects
            cluster_assignments: {cluster: [subject_id]}
        """
        images = {}
        for cluster in cluster_assignments:
            meta = {}
            cluster_subjects = subject_storage.get_subjects(
                cluster_assignments[cluster])
            print(cluster, cluster_subjects)
            # self.add_subjects(cluster, cluster_subjects, meta)
            
            for image_subjects in self.split_subjects(cluster_subjects):
                image_id = next_id()
                images[image_id] = Image(
                    image_id=image_id,
                    group_id=group_id,
                    cluster=cluster,
                    subjects=image_subjects,
                    metadata=meta)
        return images


    # def create_subjects(self, cluster, subjects, meta):
        # for image_subjects in self.split_subjects(subjects):
            # id_ = next_id()
            # if id_ in self.images:
                # raise KeyError('image id {} already exists in group {}' \
                    # .format(id_, self.group_id))
            # self.images[id_] = Image(
                # id_, self.group_id, cluster, image_subjects, meta)

    def metadata(self):
        return {
            'image_size': self.image_size,
            'image_width': self.image_width,
            'group_id': self.group_id,
            'description': self.description,
            'permutations': self.permutations
        }

    def __str__(self):
        s = 'group %s images %d metadata %s' % \
            (str(self.group_id), len(self.images), self.metadata())
        return s

    def __repr__(self):
        return str(self)

    def get_zoo(self, zoo_id):
        if self.zoo_map is None:
            zoo_map = {}
            for i in self.iter():
                if i.zoo_id:
                    zoo_map[i.zoo_id] = i.id
            self.zoo_map = zoo_map
        return self.images[self.zoo_map[zoo_id]]

    def iter(self):
        for image_id in sorted(self.images):
            yield self.images[image_id]

    def list(self):
        return list(self.iter())

    def split_subjects(self, subjects):
        """
        Subdivide a list of subjects into image groups, each of size
        determined in constructor call.
        """
        images = []

        for _ in range(self.permutations):
            keys = subjects.keys()
            random.shuffle(keys)

            # Sometimes the number of subjects in a cluster cannot be evenly
            # split into N even sized images. In this case we add enough
            # duplicate subjects to the last image to make it the same size
            if len(keys) % self.image_size > 0:
                print(len(keys), self.image_size, len(keys) % self.image_size,
                        self.image_size - (len(keys) % self.image_size))

                diff = self.image_size - (len(keys) % self.image_size)
                print(len(keys), self.image_size, len(keys) % self.image_size,
                        diff)
                keys += random.sample(keys, diff)

            length = len(keys)
            print(length)
            w = math.ceil(length/self.image_size)
            for n in range(w):
                a = n*self.image_size
                b = min(length, a+self.image_size)
                subset = keys[a:b]

                images.append(subset)
        return images

    # def remove_images(self, images):
        # for image in self.iter():
            # if image.id in images:
                # image.metadata['deleted'] = True

    def upload_subjects(self, path):
        """
        Upload generated images to Panoptes
        """
        uploader = panoptes.Uploader(muon.config.project, self.group_id)
        existing_subjects = uploader.get_subjects()
        existing_subjects = {k: v for v, k in existing_subjects}

        print('Creating Panoptes subjects')
        for image in self.iter():
            # Skip images that are already uploaded and linked to the
            # subject set, and make sure the zoo_id map is correct
            if image.image_id in existing_subjects:
                image.zoo_id = existing_subjects[image.image_id]
                print('Skipping %s' % image)
                continue

            fname = os.path.join(
                path, 'group_%d' % self.group_id, image.fname())

            subject = panoptes.Subject()
            subject.add_location(fname)
            subject.metadata.update(image.dump_manifest())

            subject = uploader.add_subject(subject)
            image.zoo_id = subject.id

        print('Uploading subjects')
        uploader.upload()

    def generate_images(self, subject_storage, path=None):
        """
        Generate subject images to be uploaded to Panoptes
        """
        path = os.path.join(path, 'group_%d' % self.group_id)
        if not os.path.isdir(path):
            os.mkdir(path)
        i = 0
        for image in self.iter():
            print(image)
            image.plot(self.image_width, subject_storage, path)
            i += 1
            if i > 5:
                break


class ImageStorage:

    def __init__(self, database):
        self.database = database
        self._groups = {}

    @property
    def conn(self):
        return self.database.conn

    def next_group(self):
        with self.conn as conn:
            return self.database.ImageGroup.next_id(conn)

    def next_id(self):
        with self.conn as conn:
            return self.database.Image.next_id(conn)

    def next_id_callback(self):
        next_id = self.next_id()
        def f():
            nonlocal next_id
            i = next_id
            next_id += 1
            return i
        return f

    def new_group(self, subjects, image_size, cluster_name,
                  batch=None, **kwargs):
        """
        Generate a file detailing which subjects belong in which image
        and their location in the image.

        """
        # images = {}
        # i = self.next_id

        # subjects = subjects.list()
        # l = len(subjects)
        # w = math.ceil(l/image_size)

        # for n in range(w):
            # a = n * image_size
            # b = min(l, a + image_size)
            # subset = subjects[a:b]

            # images[i] = Image(i, group, subset, None)

            # i += 1
        group_id = self.next_group()
        next_id = self.next_id_callback()

        with self.conn as conn:
            cluster_assignments = self.database.Clustering \
                .get_cluster_assignments(conn, cluster_name, batch)
        if not cluster_assignments:
            logger.warn('cluster_assignments: {}', cluster_assignments)
            raise Exception('Empty cluster assignment struct')

        group = ImageGroup.new(
            group_id, next_id, subjects, cluster_assignments,
            image_size=image_size, **kwargs)

        self.add_group(group)
        return group

    # @classmethod
    # def new(cls, fname):
    #     with h5py.File(fname, 'w') as f:
    #         f.attrs['next_id'] = 0
    #         f.attrs['next_group'] = 0

    #     return cls(fname)

    def add_group(self, group):
        with self.conn as conn:
            self.database.ImageGroup.add_group(conn, group)
            for image in group.iter():
                self.database.Image.add_image(conn, image)

            conn.commit()
        self._groups[group] = group

    def get_group(self, group):
        if group not in self._groups:
            self._groups[group] = self.load_group(group)
        return self._groups[group]

    def load_group(self, group_id):
        with self.conn as conn:
            group = self.database.ImageGroup.get_group(conn, group_id)
            images = self.database.Image.get_group_images(conn, group_id)

            images = {image.image_id: image for image in tqdm(images)}
            group.images = images
            return group

    def list_groups(self):
        with self.conn as conn:
            self.database.ImageGroup.list_groups(conn)

    def update_group(self, group_id):
        group = self.get_group(group_id)

        with self.conn as conn:
            for image in group.iter():
                self.database.Image.update_image(conn, image)
            conn.commit()

    def save(self, groups=None):
        # TODO somehow manage upating the database
        pass
        # with self.conn as conn:
            # groups = groups or list(self._groups)
            # print(groups)
            # for group in groups:
                # self._groups[group].dump(conn)
            # conn.commit()

#         with h5py.File(self.fname, 'r+') as f:
#             f.attrs['next_id'] = self.next_id
#             f.attrs['next_group'] = self.next_group

#             if groups is None:
#                 groups = list(self._groups)
#             for group in groups:
#                 g = 'groups/group_{}'.format(group)
#                 if g in f:
#                     del f[g]
#                 self._groups[group].dump_hdf(f.create_group(g))
