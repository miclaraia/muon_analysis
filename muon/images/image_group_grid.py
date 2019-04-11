import math
import os
import random
import logging
from tqdm import tqdm

from muon.images.image import Image
from muon.images.image_group_parent import ImageGroupParent

logger = logging.getLogger(__name__)


class GridImageGroup(ImageGroupParent):

    TYPE = ImageGroupParent.TYPES['grid']

    @classmethod
    def new(cls, database, cluster_name, cluster_assignments,
            group_id=None, **kwargs):
        """
        Parameters:
            cluster_assignments: {cluster: [subject_id]}
        """

        attrs = {
            'image_size': kwargs.get('image_size', 36),
            'group_type': cls.TYPE,
            'image_width': kwargs.get('image_width', 6),
            'description': kwargs.get('description', None),
            'permutations': kwargs.get('permutations', 1),
            'cluster_name': cluster_name,
        }

        group = super().new(database, group_id, **attrs)
        group._create_images(cluster_assignments)

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

    def _create_images(self, cluster_assignments):
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
