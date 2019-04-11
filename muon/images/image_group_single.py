from muon.images.image_group_parent import ImageGroupParent
from muon.images.single_image import SingleImage


class SingleImageGroup(ImageGroupParent):

    TYPE = ImageGroupParent.TYPES['single']

    @classmethod
    def new(cls, database, subject_ids, group_id=None, **kwargs):
        """
        Parameters:
            cluster_assignments: {cluster: [subject_id]}
        """

        attrs = {
            'image_size': 1,
            'group_type': cls.TYPE,
            'image_width': None,
            'description': kwargs.get('description', None),
            'permutations': None,
            'cluster_name': 'single',
        }

        group = super().new(database, group_id, **attrs)
        group._create_images(subject_ids)

        return group

    def _create_images(self, subject_ids):
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

        attrs = {
            'group_id': self.group_id,
            'metadata': {},
            'commit': False,
        }

        with self.database.conn as conn:
            count = 0
            for subject_id in subject_ids:
                image = SingleImage.new(
                    self.database,
                    image_id=next_id(),
                    subject=subject_id,
                    **attrs)
                self.database.Image.add_image(conn, image)
                count += 1

            self.image_count += count
            conn.commit()
        self.save()
