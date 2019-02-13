#!/usr/bin/env python

from muon.subjects.database import Database
from muon.subjects.images import ImageStorage, ImageGroup
import muon.project.panoptes as pan
import muon.config

import click
import os
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(logger.level, logging.DEBUG)
logger.debug('test2')
logging.warn('test')

@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('image_path')
def main(database_file, image_path):
    muon.config.project = 1815

    pan.Uploader._client = pan.Panoptes.connect(
        login='interactive',
        endpoint='https://panoptes-staging.zooniverse.org')

    if not os.path.isdir(image_path):
        raise FileNotFoundError(image_path)


    ids = [9, 10, 11, 12, 13]

    database = Database(database_file)
    image_storage = ImageStorage(database)

    # new_group = image_storage.get_group(100)
    # with database.conn as conn:
        # database.ImageGroup.delete_group(conn, 100)
        # conn.commit()

    image_group = image_storage.get_group(0)
    image_group.images = {i: image_group.images[i] for i in ids}
    image_group.upload_subjects(image_path)

    import code
    code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()

