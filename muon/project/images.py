import numpy as np
import math


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

        return cls(kwargs)


class Images:
    _image = Image

    def __init__(self, cluster, save_dir=None, **kwargs):
        self.subjects = subjects
        self.images = None
        # TODO load existing structure to not duplicate ids
        self.next_id = 0

        self.save_dir = save_dir

        self.size = kwargs.get('image_size', 40)
        self.image_dim = kwargs.get('image_dim', (50, 50))

    @property
    def structure(self):
        return self.generate_structure()

    def generate_structure(self):
        """
        Generate a file detailing which subjects belong in which image
        and their location in the image.

        """
        images = []
        i = self.next_id

        subjects = self.subjects.list()
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


    def save_structure(self, images):
        data = {}
        data['images'] = [i.dump() for i in images]

        # TODO save to different file per upload...? Or have them all in the
        # same file. Probably want them all in the same file.
        fname = os.path.join(self.save_dir, 'structure.json')
        with open(fname, 'w') as file:
            json.dump(data, file)


    def load_structure(self):
        fname = os.path.join(self.save_dir, 'structure.json')
        with open(fname, 'r') as file:
            data = json.load(file)

        I = self._image
        images = []

        


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

