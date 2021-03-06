
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import logging

from muon.subjects.subjects import Subjects
from muon.database.utils import StorageObject, StorageAttribute, \
        StoredAttribute
from muon.images import TYPES

logger = logging.getLogger(__name__)


class ImageParent(StorageObject):
    """
    A group of subjects which are uploaded to Panoptes as a single Image
    """

    cluster = StorageAttribute('cluster')
    metadata = StorageAttribute('metadata')
    image_meta = StorageAttribute('image_meta')
    zoo_id = StorageAttribute('zoo_id')
    subjects = StorageAttribute('subjects')
    TYPES = TYPES

    def __init__(self, image_id, database, attrs=None, online=False):
        super().__init__(database, online)
        self.image_id = image_id

        if attrs is None:
            with self.conn as conn:
                # TODO need to change database method
                attrs = database.Image.get_image(conn, image_id)
        subjects = attrs.get('subjects')
        if attrs['image_meta']:
            image_meta = self.ImageMeta(**attrs['image_meta'])
        else:
            image_meta = None

        self.group_id = attrs['group_id']

        storage = [
            StoredAttribute('cluster', attrs['cluster']),
            StoredAttribute('metadata', attrs['metadata']),
            StoredAttribute('zoo_id', attrs['zoo_id']),
            StoredAttribute('image_meta', image_meta),
            SubjectLoader('subjects', image_id, database, subjects=subjects)]
        self.storage = {s.name: s for s in storage}

    @classmethod
    def new(cls, database, group_id, cluster, metadata, subjects,
            image_id=None, zoo_id=None, image_meta=None, commit=True):

        if image_id is None:
            with database.conn as conn:
                image_id = database.Image.next_id(conn)

        attrs = {
            'group_id': group_id,
            'cluster': cluster,
            'metadata': metadata,
            'subjects': subjects,
            'zoo_id': zoo_id,
            'image_meta': image_meta
        }

        image = cls(image_id, database, attrs=attrs)
        if commit:
            with database.conn as conn:
                database.Image.add_image(conn, image)
                conn.commit()

        return image

    def save(self):
        updates = {}
        for k, v in self.storage.items():
            if v.has_changed:
                if k == 'image_meta':
                    updates.update(v.value.dump_db())
                else:
                    updates[k] = v.value
                v.has_changed = False

        if updates:
            with self.conn as conn:
                self.database.Image.update_image(conn, self.image_id, updates)
                conn.commit()

    def __str__(self):
        return 'id {} group {} cluster {} ' \
               'metadata {} zooid {} image_meta {}'.format(
                    self.image_id, self.group_id, self.cluster,
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
        return self._fname_static(self.image_id, self.group_id)

    @staticmethod
    def _fname_static(image_id, group_id):
        """
        Filename to use for this subject group
        """
        return 'muon_group_%d_id_%d.jpg' % (group_id, image_id)

    def generate(self, width, subject_storage, dpi=None, quality=None,
                 path=None):
        """
        Generate and save a plot of this image
        """
        fname = self.fname()
        # Skip if image already exists
        if fname in os.listdir(path):
            return False
        if path:
            fname = os.path.join(path, fname)
        # Touch the file
        # so other concurrent image generation processes
        # skip this image

        try:
            open(fname, 'w').close()

            if quality is None:
                quality = 95

            dpi = dpi or 50
            fig, meta = self.plot(width, subject_storage)

            self.image_meta = self.__class__ \
                .ImageMeta(dpi=dpi, **meta)
            # TODO make sure to actually update this data in the db
            # Probably need to add new columns

            fig.savefig(fname, dpi=dpi, quality=quality)

            plt.close(fig)
            return True

        except (KeyboardInterrupt, Exception):
            os.remove(fname)
            raise

    def plot(self, width, subject_storage, fig=None, classifications=None):
        subjects = []
        keys = set()
        for s in self.subjects:
            subject = subject_storage.get_subject(s)
            if subject.id in keys:
                subject.id = subject.id+'-duplicate'
            keys.add(subject.id)
            subjects.append(subject)

        subjects = Subjects(subjects)

        offset = .5
        fig, meta = Subjects(subjects).plot_subjects(
            w=width, grid=True, grid_args={'offset': offset}, meta=True,
            classifications=classifications)

        meta['offset'] = offset
        return fig, meta

    def at_location(self, x, y):
        """
        Return the subject that should be at the given x,y coordinates
        """
        logger.debug(self)
        logger.debug(self.metadata)
        logger.debug(self.image_meta)

        dpi = self.image_meta.dpi
        offset = self.image_meta.offset*dpi
        height = self.image_meta.height*dpi - offset
        width = self.image_meta.width*dpi - offset

        rows = self.image_meta.rows
        cols = self.image_meta.cols

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

        if x >= cols or y >= rows:
            # Selection outside of bounds
            logger.error('Selection outside of bounds of image')
            logger.warn(self)
            logger.warn((x, y))
            return

        i = int(x+cols*y)
        logger.debug(i)
        if i >= len(self.subjects):
            logger.error('Index out of bounds')
            logger.error(i)
            logger.error(self)
            logger.error(self.subjects)
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

        def dump_db(self):
            return {
                'fig_dpi': self.dpi,
                'fig_offset': self.offset,
                'fig_height': self.height,
                'fig_width': self.width,
                'fig_rows': self.rows,
                'fig_cols': self.cols
            }
        def __str__(self):
            return str(self.__dict__)


class SubjectLoader(StoredAttribute):

    def __init__(self, name, image_id, database, subjects=None):
        super().__init__(name, subjects)

        self.image_id = image_id
        self.database = database

    def get(self):
        if self.value is None:
            self.value = self._load()
        return self.value

    def _load(self):
        # TODO load subjects from database
        with self.database.conn as conn:
            logger.debug('image %s getting subjects', self.image_id)
            subjects = self.database.Image \
                .get_image_subjects(conn, self.image_id)
            return list(subjects)
