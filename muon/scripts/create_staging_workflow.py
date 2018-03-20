#!/usr/bin/env python

from muon.project.images import Images
import muon.project.panoptes as pan

import click

@click.group(invoke_without_command=True)
@click.argument('path')
def main(path):
    ids = [1282, 1285, 1287, 1168, 1035, 1037, 1298, 1299, 1241, 792]
    images1 = Images.load_group(2)
    images = {i: images1.images[i] for i in ids}
    images = Images(3, images, images1.next_id,
                    description='Images uploaded to panoptes staging project')
    print(path)

    pan.Uploader._client = pan.Panoptes(
        login='interactive',
        endpoint='https://panoptes-staging.zooniverse.org')

    images.upload_subjects(path)
    

    


if __name__ == '__main__':
    main()

