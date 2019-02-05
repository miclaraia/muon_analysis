import math
import os
import json
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import csv
import logging
import sqlite3

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
                 cluster, subjects, metadata, zoo_id=None):
        self.image_id = image_id
        self.group_id = group_id
        self.cluster = cluster
        self.subjects = subjects
        self.metadata = metadata
        self.zoo_id = zoo_id

    @classmethod
    def load(cls, conn, image_id):
        cursor = conn.execute(
            'SELECT * FROM images WHERE image_id=?', (image_id,))
        row = cursor.fetchone()
        fields = [
            'image_id',
            'group_id',
            'cluster',
            'metadata',
            'zoo_id']
        kwargs = {f: row[i] for i, f in enumerate(fields)}
        kwargs['metadata'] = json.loads(kwargs['metadata'])

        cursor = conn.execute(
            'SELECT * FROM subjects WHERE image_id=?', (image_id,))
        rows = cursor.fetchall()
        rows = sorted(rows, key=lambda row: row[3])
        
        subjects = [row[0] for row in rows]
        return cls(subjects=subjects, **kwargs)

    def dump(self, conn):
        print('dumping', str(self))
        conn.execute(
            'INSERT INTO images VALUES (?,?,?,?,?)', (
                self.image_id,
                self.group_id,
                self.cluster,
                json.dumps(self.metadata),
                self.zoo_id)
        )

        args = []
        for i in range(len(self.subjects)):
            subject = self.subjects[i]
            args.append((subject, self.image_id, self.group_id, i))
            #metadata = {'index': i}

        # args = [(self.image_id, subject, i) for i in range(len(self.subjects))]
        conn.executemany('INSERT INTO subjects VALUES (?,?,?,?)', args)

    def __str__(self):
        return 'id {} group {} cluster {} subjects {} ' \
               'metadata {} zooid {}'.format(
            self.image_id, self.group_id, self.cluster, len(self.subjects),
            str(self.metadata), self.zoo_id)

    def __repr__(self):
        return str(self)

    # def dump(self):
        # return {
            # 'id': self.id,
            # 'group': self.group,
            # 'subjects': [int(i) for i in self.subjects],
            # 'metadata': self.metadata,
            # 'zooniverse_id': self.zoo_id
        # }

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

        metadata = {
            'dpi': dpi,
            'offset': offset,
            **meta
        }
        self.metadata.update({'figure': metadata})

        fig.savefig(fname, dpi=dpi)

        plt.close(fig)

    def at_location(self, x, y):
        """
        Return the subject that should be at the given x,y coordinates
        """
        meta = self.metadata['figure']
        logger.debug(meta)
        dpi = meta['dpi']
        offset = meta['offset']*dpi
        height = meta['height']*dpi-offset
        width = meta['width']*dpi-offset

        if 'beta_image' in meta:
            # This image was generated before the offset bug was discovered
            # and need to correct the vertical offset to get the right
            # boundary calculations
            width = meta['width']*dpi*0.97
            height = meta['height']*dpi*0.97
            y = y - 0.03*meta['height']*dpi + offset

        y_ = height/meta['rows']
        x_ = width/meta['cols']
        logger.debug('x: {}'.format(x))
        logger.debug('y: {}'.format(y))
        logger.debug('offset: {}'.format(offset))
        logger.debug('width: {}'.format(width))
        logger.debug('height: {}'.format(height))

        logger.debug('x_: {}'.format(x_))
        logger.debug('y_: {}'.format(y_))

        x = (x-offset)//x_
        y = (y-offset)//y_

        i = int(x+meta['cols']*y)
        logger.debug(i)
        return self.subjects[i]


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
    def load(cls, conn, group):
        cursor = conn.execute('SELECT * FROM groups WHERE group_id=?',
                              (group,))
        row = cursor.fetchone()

        fields = [
            'group_id',
            'image_size',
            'image_width',
            'description',
            'permutations'
        ]
        kwargs = {f: row[i] for i, f in enumerate(fields)}

        cursor = conn.execute('SELECT image_id FROM images WHERE group_id=?',
                              (kwargs['group_id'],))
        images = [Image.load(conn, row[0]) for row in cursor]
        images = {image.image_id: image for image in images}

        return cls(images=images, **kwargs)

    def clear(self, conn):
        print(self.group_id, type(self.group_id))
        conn.execute('DELETE FROM groups WHERE group_id=?', (self.group_id,))
        conn.execute('DELETE FROM images WHERE group_id=?', (self.group_id,))
        conn.execute('DELETE FROM subjects WHERE group_id=?', (self.group_id,))

    def dump(self, conn):
        self.clear(conn)
        print('dumping', str(self))
        query = 'INSERT INTO groups VALUES (?,?,?,?,?)'
        args = (self.group_id,
                self.image_size,
                self.image_width,
                self.description,
                self.permutations)
        print(query, args)
        conn.execute(query, args)

        # have each image dump to sql as well
        for i in self.images:
            self.images[i].dump(conn)


    @classmethod
    def new(cls, group, next_id,
            subject_storage, cluster_assignments, **kwargs):
        """
        Parameters:
            next_id: callback function to get next image id
            cluster_assignments: {cluster: [subject_id]}
        """
        self = cls(group, None, **kwargs)
        self.add_clustered_subjects(
            next_id, subject_storage, cluster_assignments)
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

    def add_clustered_subjects(
            self, next_id, subject_storage, cluster_assignments):
        """
        Parameters:
            next_id: next id for a new image
            subjects
            cluster_assignments: {cluster: [subject_id]}
        """
        for c in cluster_assignments:
            meta = {}
            cluster_subjects = subject_storage.get_subjects(cluster_assignments[c])
            self.add_subjects(next_id, c, cluster_subjects, meta)

    def add_subjects(self, next_id, cluster, subjects, meta):
        for image_subjects in self.split_subjects(subjects):
            id_ = next_id()
            if id_ in self.images:
                raise KeyError('image id {} already exists in group {}' \
                    .format(id_, self.group_id))
            self.images[id_] = Image(
                id_, self.group_id, cluster, image_subjects, meta)

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
        for image in self.list():
            yield image

    def list(self):
        return list(self.images.values())

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

    def remove_images(self, images):
        for image in self.iter():
            if image.id in images:
                image.metadata['deleted'] = True

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
            if image.id in existing_subjects:
                image.zoo_id = existing_subjects[image.id]
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
        for image in self.iter():
            print(image)
            image.plot(self.image_width, subject_storage, path)


class SQLImages:

    def __init__(self, fname):
        self.fname = fname
        self.next_id = 0
        self.next_group = 0
        self._groups = {}

        if not os.path.isfile(fname):
            self.create_db()

    @property
    def conn(self):
        return self.__class__.Connection(self.fname)
        # return sqlite3.connect(self.fname)

    def create_db(self):
        with self.conn as conn:

#         Tables:
#             Images
#                 image_id, group_id, metadata, zoo_id
#             ImageSubjects
#                 subject_id, image_id, image_location details
#             ImageGroups
#                 group_id, metadata->(size, dim, group, description)
            query = ''
            query = """
                CREATE TABLE IF NOT EXISTS images (
                    image_id integer PIMARY KEY,
                    group_id integer,
                    cluster integer,
                    metadata text NOT NULL,
                    zoo_id integer
                );

                CREATE TABLE IF NOT EXISTS subjects (
                    subject_id integer,
                    image_id integer,
                    group_id integer,
                    image_index integer
                );

                CREATE TABLE IF NOT EXISTS groups (
                    group_id integer PRIMARY KEY,
                    image_size integer,
                    image_width integer,
                    description text,
                    permutations integer
                );
            """
            print(query)
            conn.executescript(query)

    def new_group(self, subjects, cluster_assignments, image_size, **kwargs):
        """
        Generate a file detailing which subjects belong in which image
        and their location in the image.

        """
        # images = {}
        # i = self.next_id
        group = int(self.next_group)

        # subjects = subjects.list()
        # l = len(subjects)
        # w = math.ceil(l/image_size)

        # for n in range(w):
            # a = n * image_size
            # b = min(l, a + image_size)
            # subset = subjects[a:b]

            # images[i] = Image(i, group, subset, None)

            # i += 1
        image_group = ImageGroup.new(
            group, self.next_id_callback(), subjects, cluster_assignments,
            image_size=image_size, **kwargs)
        # self.next_id = i

        # image_group = ImageGroup(group, images, image_size=image_size, **kwargs)
        self._groups[group] = image_group
        self.save()

    # @classmethod
    # def new(cls, fname):
    #     with h5py.File(fname, 'w') as f:
    #         f.attrs['next_id'] = 0
    #         f.attrs['next_group'] = 0

    #     return cls(fname)

    def next_id_callback(self):
        def callback():
            i = self.next_id
            self.next_id += 1
            return i
        return callback

    def get_group(self, group):
        if group not in self._groups:
            self._groups[group] = self.load_group(group)
        return self._groups[group]

    def load_metadata(self):
        with self.conn as conn:
            cursor = conn.execute('SELECT MAX(image_id) FROM images')
            #try:
            self.next_id = cursor.fetchone[0] + 1
            #except

            cursor = conn.execute('SELECT MAX(group_id) FROM groups')
            self.next_group = cursor.fetchone()[0] + 1

    def load_group(self, group):
        with self.conn as conn:
            return ImageGroup.load(conn, group)

    def list_groups(self):
        with self.conn as conn:
            cursor = conn.execute('SELECT group_id FROM groups')
            return [row[0] for row in cursor]

    def save(self, groups=None):
        with self.conn as conn:
            groups = groups or list(self._groups)
            print(groups)
            for group in groups:
                self._groups[group].dump(conn)
            conn.commit()

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

    class Connection:

        def __init__(self, fname):
            self.fname = fname
            self.conn = None

        def __enter__(self):
            self.conn = sqlite3.connect(self.fname)
            return self.conn

        def __exit__(self, type, value, traceback):
            self.conn.close()




import click

@click.group()
def cli():
    pass

@cli.command()
def main():
    fname = '/mnt/storage/science_data/muon_data/zooniverse/MH2/' \
            'image_structure/data.h5'
    HDFImages.new(fname)


@cli.command()
@click.argument('fname')
@click.argument('group')
def load(fname, group):
    images = HDFImages(fname)
    _group = images.get_group(0)

    import code
    code.interact(local={**globals(), **locals()})


if __name__ ==  '__main__':
    cli()
