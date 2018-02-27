#!/usr/bin/env python

from muon.project.images import Images
import math


def main():
    for g in [1, 2]:
        images = Images.load_group(g)
        for image in images.iter():
            N = len(image.subjects)
            cols = 10
            rows = math.ceil(N/10)
            image.metadata.update({
                'figure': {
                    'cols': cols,
                    'rows': rows,
                    'height': 2*rows,
                    'width': 2*cols,
                    'dpi': 100,
                    'offset': 0.03*(2*cols),
                    'beta_image': True,
                }
            })

        images.save_group(overwrite=True)


if __name__ == '__main__':
    main()
