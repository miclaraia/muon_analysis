import numpy as np
import math
import os
import json
from shutil import copyfile

import muon.data


class Image:

    def __init__(self, id_, subjects, metadata, zoo_id=None):
        self.id = id_
        self.subjects = subjects
        self.metadata = metadata
        self.zoo_id = zoo_id

    def dump(self):
        return {
            'id': self.id,
            'subjects': self.subjects,
            'metadata': self.metadata,
            'zooniverse_id': self.zoo_id
        }

    @classmethod
    def load(cls, dumped):
        kwargs = {
            'id_': dumped['id'],
            'subjects': dumped['subjects'],
            'metadata': dumped['metadata'],
            'zoo_id': dumped['zooniverse_id'],
        }

        return cls(**kwargs)


class Images:
    _image = Image

    def __init__(self, group, images, next_id, **kwargs):
        self.images = images
        # TODO load existing structure to not duplicate ids
        self.next_id = next_id
        self.group = group

        self.size = kwargs.get('image_size', 40)
        self.image_dim = kwargs.get('image_dim', (50, 50))
        self.description = kwargs.get('description', None)

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
        data = data['groups'][group]

        images = [i.load() for i in data['images']]
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
        return muon.data.path('structure.json')

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

            images.append(Image(i, subset, None))

            i += 1
        self.next_id = i

        return images

    def save_group(self, images):
        fname = self._fname()
        if os.path.isfile(fname):
            with open(fname, 'r') as file:
                data = json.load(file)

            copyfile(fname, fname+'.bak')
        else:
            data = {'groups': {}}

        if self.group in data['groups']:
            print('file contents: ', data)
            raise Exception('Refusing to overwrite group (%d) in structure '
                            'file' % self.group)

        data['groups'][self.group] = {
            'metadata': self.metadata(),
            'images': [i.dump() for i in images]
        }
        data['next_id'] = self.next_id
        data['next_group'] = self.group + 1

        # TODO save to different file per upload...? Or have them all in the
        # same file. Probably want them all in the same file.
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
        pass

    def generate_images(self):
        """
        Generate subject images to be uploaded to Panoptes
        """
        pass


class Random_Images(Images):

    def __init__(self, subjects, **kwargs):
        super().__init__(subjects, **kwargs)

    def _structure(self):
        pass

