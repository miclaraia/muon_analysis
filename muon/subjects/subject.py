import numpy as np
import math

from muon.utils.camera import Camera, CameraPlot, CameraRotate


class Subject:

    def __init__(self, id, charge, metadata, label=None):
        self.id = id
        self.metadata = metadata
        self.x = self._normalize(charge)
        self.y = label

        self.zoo_id = metadata.get('zoo_id')

    @property
    def event(self):
        return (self.metadata['run'],
                self.metadata['evt'],
                self.metadata['tel'])

    def plot(self, ax, camera=None):
        if camera is None:
            camera = Camera()

        data = camera.transform(self.x, False)
        CameraPlot.plot(data, ax, radius=camera.pixSideLength)
        return ax

    @staticmethod
    def color():

        return (.9, .1, .1)

    @staticmethod
    def _normalize(charge):
        charge = np.array(charge)
        n = np.linalg.norm(charge)
        d = charge.shape[0]
        if n == 0:
            return charge.copy()
        else:
            return charge / n * math.sqrt(d)

    def rotations(self):
        for n in range(6):
            yield CameraRotate().rotate(self.x, n)

    def __str__(self):
        return 'id %d event %s' % \
               (self.id, self.event)

    def copy(self, n=0):
        x = CameraRotate().rotate(self.x, n)
        return self.__class__(self.id, x, self.metadata, self.y)

