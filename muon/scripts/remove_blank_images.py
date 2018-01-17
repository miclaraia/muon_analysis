#!/usr/bin/env python

import muon.project.panoptes as pan
from muon.project.images import Images, Random_Images


def main():
    images = Images.load_group(0)

    to_remove = []
    for image in images.images:
        if image.metadata['cluster'] == 0:
            to_remove.append(image)

    for i in to_remove:
        print(i)

    images.remove_images([i.id for i in to_remove])

    uploader = pan.Uploader(5918, images.group)
    uploader.unlink_subjects([i.zoo_id for i in to_remove])
    images.save_group(True)

    import code
    code.interact(local=locals())


if __name__ == '__main__':
    main()

