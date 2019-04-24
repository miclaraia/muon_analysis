from muon.images.image_parent import ImageParent
from muon.images.single_image import SingleImage
from muon.images.image_group_parent import ImageGroupParent


class Image(ImageParent):

    @classmethod
    def image_type(cls, image_type):
        if image_type == cls.TYPES['grid']:
            return cls
        elif image_type == cls.TYPES['single']:
            return SingleImage

    @classmethod
    def create_image(cls, image_type, *args, **kwargs):
        return cls.image_type(image_type)(*args, **kwargs)
