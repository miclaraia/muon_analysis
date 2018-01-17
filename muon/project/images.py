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

import muon.project.panoptes as panoptes

import muon.data


class Image:
    """
    A group of subjects which are uploaded to Panoptes as a single Image
    """

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

    def dump_manifest(self):
        data = OrderedDict([
            ('id', self.id),
            ('#group', self.group),
            ('image', self.fname()),
            ('subjects', self.subjects)])
        for k, v in self.metadata.items():
            if k not in data:
                data[k] = v

        return data


    @classmethod
    def load(cls, dumped):
        """
        Load Image from an entry in structures file
        """
        kwargs = {
            'id_': dumped['id'],
            'group': dumped['group'],
            'subjects': dumped['subjects'],
            'metadata': dumped['metadata'],
            'zoo_id': dumped['zooniverse_id'],
        }

        return cls(**kwargs)

    def fname(self):
        """
        Filename to use for this subject group
        """
        return 'muon_group_%d_id_%d.png' % (self.group, self.id)

    def plot(self, width, subjects, path=None):
        """
        Generate and save a plot of this image
        """
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

    def at_location(self, x, y, width):
        """
        Return the subject that should be at the given x,y coordinates
        """
        x = x//200
        y = y//200

        i = x+width*y
        return self.subjects[i]


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

    def __str__(self):
        s = 'group %d images %d metadata %s' % \
            (self.group, len(self.images), self.metadata())
        return s

    def __repr__(self):
        return str(self)

    @classmethod
    def new(cls, cluster, **kwargs):
        """
        Create new Images group
        """
        group, next_id = cls.load_metadata()
        images = cls(group, None, next_id, **kwargs)
        images.generate_structure(cluster)

        return images

    @classmethod
    def load_group(cls, group):
        """
        Load Images object from group entry in structures json file
        """
        fname = cls._fname()
        if os.path.isfile(fname):
            with open(fname, 'r') as file:
                data = json.load(file)

        next_id = data['next_id']
        data = data['groups'][str(group)]

        images = []
        for item in data['images']:
            image = cls._image.load(item)
            if image.metadata.get('deleted') is True:
                continue
            images.append(image)

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
        """
        Subdivide a list of subjects into image groups, each of size
        determined in constructor call.
        """
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

    def remove_images(self, images):
        for image in self.images:
            if image.id in images:
                image.metadata['deleted'] = True

    def save_group(self, overwrite=False, backup=None):
        """
        Save the configuration of this Images object to the structures
        json file
        """
        images = self.images
        group = str(self.group)

        fname = self._fname()
        if os.path.isfile(fname):
            with open(fname, 'r') as file:
                data = json.load(file)

            if backup is None:
                backup = fname+'.bak'
            copyfile(fname, backup)
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
        """
        Load metadata stored in the header of the structures json file
        """
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

    def upload_subjects(self, path):
        """
        Upload generated images to Panoptes
        """
        uploader = panoptes.Uploader(5918, self.group)
        existing_subjects = uploader.get_subjects()
        existing_subjects = {k: v for v, k in existing_subjects}

        print('Creating Panoptes subjects')
        for image in self.images:
            # Skip images that are already uploaded and linked to the
            # subject set, and make sure the zoo_id map is correct
            if image.id in existing_subjects:
                image.zoo_id = existing_subjects[image.id]
                print('Skipping %s' % image)
                continue

            fname = os.path.join(path, image.fname())

            subject = panoptes.Subject()
            subject.add_location(fname)
            subject.metadata.update(image.dump_manifest())

            subject = uploader.add_subject(subject)
            image.zoo_id = subject.id

        print('Uploading subjects')
        uploader.upload()
        self.save_group(True)

    def generate_manifest(self):
        """
        Generate the subject manifest for Panoptes
        """
        raise DeprecationWarning
        fname = muon.data.path('subject_manifest_%d' % self.group)
        keys = list(self.images[0].dump_manifest().keys())

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
    """
    Creates images of subjects grouped by machine learning cluster.
    Subjects are randomly shuffled within each cluster.
    """

    def generate_structure(self, cluster):
        subjects = cluster.subjects

        images = []
        i = self.next_id

        for c in range(cluster.config.n_clusters):
            # Skip empty subjects
            if c == 0:
                continue

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
