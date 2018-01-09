import numpy as np
import math
import os
import json
from shutil import copyfile
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import csv
import socket

import muon.data


class Image:

    def __init__(self, id_, group, subjects, metadata, zoo_id=None):
        self.id = id_
        self.group = group
        self.subjects = subjects
        self.metadata = metadata
        self.zoo_id = zoo_id

    def __str__(self):
        return 'id %d group %d subjects %d metadata %s zooid %s' % \
                (self.id, self.group, len(self.subjects),
                 str(self.metadata), self.zoo_id)

    def __repr__(self):
        return str(self)

    def dump(self):
        return {
            'id': self.id,
            'group': self.group,
            'subjects': [int(i) for i in self.subjects],
            'metadata': self.metadata,
            'zooniverse_id': self.zoo_id
        }

    def dump_manifest(self, fname):
        data = OrderedDict([
            ('id', self.id),
            ('#group', self.group),
            ('image', fname),
            ('subjects', self.subjects)])
        for k, v in self.metadata.items():
            if k not in data:
                data[k] = v

        return data


    @classmethod
    def load(cls, dumped):
        kwargs = {
            'id_': dumped['id'],
            'group': dumped['group'],
            'subjects': dumped['subjects'],
            'metadata': dumped['metadata'],
            'zoo_id': dumped['zooniverse_id'],
        }

        return cls(**kwargs)

    def fname(self):
        return 'muon_group_%d_id_%d.png' % (self.group, self.id)

    def plot(self, width, subjects, path=None):
        subjects = subjects.subset(self.subjects)
        fname = self.fname()

        # Skip if image already exists
        if fname in os.listdir(path):
            return

        if path:
            fname = os.path.join(path, fname)

        fig = subjects.plot_subjects(w=width)
        fig.savefig(fname)
        plt.close(fig)
        

class Images:
    _image = Image

    def __init__(self, group, images, next_id, **kwargs):
        self.images = images
        # TODO load existing structure to not duplicate ids
        self.next_id = next_id
        self.group = group

        self.size = kwargs.get('image_size', 40)
        self.image_dim = kwargs.get('width', 10)
        self.description = kwargs.get('description', None)
        self.permutations = kwargs.get('permutations', 3)

    @classmethod
    def new(cls, cluster, **kwargs):
        group, next_id = cls.load_metadata()
        images = cls(group, None, next_id, **kwargs)
        images.generate_structure(cluster)

        return images

    @classmethod
    def load_group(cls, group):
        fname = cls._fname()
        if os.path.isfile(fname):
            with open(fname, 'r') as file:
                data = json.load(file)

        next_id = data['next_id']
        data = data['groups'][str(group)]

        images = [cls._image.load(i) for i in data['images']]
        images = cls(group, images, next_id)

        images.metadata(data['metadata'])
        return images

    def metadata(self, new=None):
        if new is None:
            return {
                'size': self.size,
                'dim': self.image_dim,
                'group': self.group,
                'description': self.description,
            }
        else:
            self.size = new['size']
            self.image_dim = new['dim']
            self.group = new['group']
            self.description = new['description']

    @staticmethod
    def _fname():
        fname = '%s_structure.json' % socket.gethostname()
        return muon.data.path(fname)

    def generate_structure(self, cluster):
        """
        Generate a file detailing which subjects belong in which image
        and their location in the image.

        """
        subjects = cluster.subjects
        images = []
        i = self.next_id

        subjects = subjects.list()
        l = len(subjects)
        w = math.ceil(l/self.size)

        for n in range(w):
            a = n * self.size
            b = min(l, a + self.size)
            subset = subjects[a:b]

            images.append(Image(i, self.group, subset, None))

            i += 1
        self.next_id = i

        self.images = images
        return images

    def split_subjects(self, subjects):
        images = []

        for _ in range(self.permutations):
            keys = subjects.keys()
            random.shuffle(keys)

            length = len(keys)
            w = math.ceil(length/self.size)
            for n in range(w):
                a = n*self.size
                b = min(length, a+self.size)
                subset = keys[a:b]

                images.append(subset)
        return images

    def save_group(self, overwrite=False):
        images = self.images
        group = str(self.group)

        fname = self._fname()
        if os.path.isfile(fname):
            with open(fname, 'r') as file:
                data = json.load(file)
            copyfile(fname, fname+'.bak')
        else:
            data = {'groups': {}}

        if group in data['groups'] and not overwrite:
            print('file contents: ', data)
            raise Exception('Refusing to overwrite group (%s) in structure '
                            'file' % group)

        data['groups'][group] = {
            'metadata': self.metadata(),
            'images': [i.dump() for i in images]
        }
        data['next_id'] = self.next_id
        data['next_group'] = self.group + 1

        # TODO save to different file per upload...? Or have them all in the
        # same file. Probably want them all in the same file.
        # TODO do we need a separate file per workflow?
        with open(fname, 'w') as file:
            json.dump(data, file)

    @classmethod
    def load_metadata(cls):
        fname = cls._fname()
        if os.path.isfile(fname):
            with open(fname, 'r') as file:
                data = json.load(file)

            next_id = data['next_id']
            group = data['next_group']
        else:
            next_id = 0
            group = 0

        return group, next_id

    def generate_manifest(self):
        """
        Generate the subject manifest for Panoptes
        """
        fname = muon.data.path('subject_manifest_%d' % self.group)
        keys = list(self.images[0].dump_manifest(None).keys())

        with open(fname, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()

            for image in self.images:
                writer.writerow(image.dump_manifest())

    def generate_images(self, subjects, path=None):
        """
        Generate subject images to be uploaded to Panoptes
        """
        for image in self.images:
            print(image)
            image.plot(self.image_dim, subjects, path)


class Random_Images(Images):

    # def __init__(self, subjects, **kwargs):
        # super().__init__(subjects, **kwargs)

    def generate_structure(self, cluster):
        subjects = cluster.subjects

        images = []
        i = self.next_id

        for c in range(cluster.config.n_clusters):
            subjects = cluster.feature_space.cluster_subjects(c)
            subsets = self.split_subjects(subjects)

            for subset in subsets:
                meta = {
                    'cluster': c,
                }
                images.append(Image(i, self.group, subset, meta))
                i += 1

        self.next_id = i
        self.images = images
        return images
