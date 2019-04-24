from collections import OrderedDict
import os
import matplotlib.pyplot as plt

from muon.images.image_parent import ImageParent


class SingleImage(ImageParent):

    @classmethod
    def new(cls, database, group_id, metadata, subject, **kwargs):
        return super().new(
            database, group_id, None, metadata, [subject], **kwargs)

    @property
    def subject(self):
        return self.subjects[0]

    def __str__(self):
        return 'solo image id {} group {} subject {} ' \
               'metadata {} zooid {} image_meta {}'.format(
                    self.image_id, self.group_id, self.subject,
                    str(self.metadata), self.zoo_id, self.image_meta)

    def __repr__(self):
        return str(self)

    def dump_manifest(self):
        data = OrderedDict([
            ('id', self.image_id),
            ('#group', self.group_id),
            ('image', self.fname()),
            ('#subject', self.subject)])
        hide = []
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
        return 's_muon_group_%d_id_%d.jpg' % (group_id, image_id)

    def plot(self, width, subject_storage, dpi=None, quality=None, path=None):
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

            subject = subject_storage.get_subject(self.subject)
            fig = plt.figure(figsize=(1, 1))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0,
                                wspace=0)
            ax = fig.add_subplot(111)
            ax.set_xticks([])
            ax.set_yticks([])
            subject.plot(ax)

            dpi = dpi or 300
            self.image_meta = ImageParent.ImageMeta(dpi=dpi)

            fig.savefig(fname, dpi=dpi, quality=quality)
            plt.close(fig)
            return True

        except (KeyboardInterrupt, Exception):
            os.remove(fname)
            raise
