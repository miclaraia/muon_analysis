
import math
import numpy as np
import matplotlib.pyplot as plt


class Camera:

    def __init__(self, length=None):
        # hexagon pixel side length
        self.pixSideLength = length or 1.
        self.numCamSpirals = 13

        self._coordinates = None

    @property
    def coordinates(self):
        if self._coordinates is None:
            raw = self.map_coordinates()
            coordinates = {}

            for i, x, y in raw:
                coordinates[i] = (x, y)

            self._coordinates = coordinates

        return self._coordinates

    def map_coordinates(self):
        deltaX = math.sqrt(3) * self.pixSideLength / 2.
        deltaY = (3. / 2. * self.pixSideLength)

        coordinates = []
        index = 1

        def add(x, y):
            nonlocal index
            coordinates.append((index, x, y))
            index += 1

        add(0, 0)

        for spiral in range(1, self.numCamSpirals + 1):

            xPos = 2.*float((spiral)) * deltaX
            yPos = 0.

            # Find the center camera location and the side length for the
            # hexagons, then use that as the initial x,y location and run
            # this algorithm to create a mapping to the real images.
            # For the two outermost spirals, there is not a pixel in
            # the y=0 row.
            if spiral < 12:
                add(xPos, yPos)

            nextPixDir = np.zeros((spiral*6, 2))
            skipPixel = np.zeros((spiral*6, 1))

            for y in range(spiral*6-1):
                # print "%d" % (y/spiral)
                if y/spiral < 1:
                    nextPixDir[y, :] = [-1, -1]
                elif y/spiral >= 1 and y/spiral < 2:
                    nextPixDir[y, :] = [-2, 0]
                elif y/spiral >= 2 and y/spiral < 3:
                    nextPixDir[y, :] = [-1, 1]
                elif y/spiral >= 3 and y/spiral < 4:
                    nextPixDir[y, :] = [1, 1]
                elif y/spiral >= 4 and y/spiral < 5:
                    nextPixDir[y, :] = [2, 0]
                elif y/spiral >= 5 and y/spiral < 6:
                    nextPixDir[y, :] = [1, -1]


            # The two outer spirals are not fully populated with pixels.
            # The second outermost spiral is missing only six pixels
            # (one was excluded above).
            if spiral == 12:
                for i in range(1, 6):
                    skipPixel[spiral*i-1] = 1
            # The outmost spiral only has a total of 36 pixels.
            # We need to skip over the
            # place holders for the rest.
            if spiral == 13:
                skipPixel[0:3] = 1
                skipPixel[9:16] = 1
                skipPixel[22:29] = 1
                skipPixel[35:42] = 1
                skipPixel[48:55] = 1
                skipPixel[61:68] = 1
                skipPixel[74:77] = 1

            for y in range(spiral*6-1):

                xPos += nextPixDir[y, 0]*deltaX
                yPos += nextPixDir[y, 1]*deltaY

                if skipPixel[y, 0] == 0:
                    add(xPos, yPos)

        return coordinates

    def plot(self):
        data = self.map_coordinates()
        _, x, y = zip(*data)
        plt.scatter(x, y)

        for i, x, y in data:
            plt.text(x, y, str(i))

        plt.axes().set_aspect('equal', 'datalim')

        plt.show()



