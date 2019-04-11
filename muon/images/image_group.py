from muon.images.image_group_parent import ImageGroupParent
from muon.images.image_group_grid import GridImageGroup
from muon.images.image_group_single import SingleImageGroup


class ImageGroup(ImageGroupParent):

    @classmethod
    def _group_type(cls, group_type):
        if group_type == cls.TYPES['grid']:
            return GridImageGroup
        elif group_type == cls.TYPES['single']:
            return SingleImageGroup

    @classmethod
    def load(cls, group_id, database, **kwargs):
        with database.conn as conn:
            attrs = database.ImageGroup.get_group(conn, group_id)

        ImageGroup_ = cls._group_type(attrs['group_type'])
        return ImageGroup_(group_id, database, **kwargs)


