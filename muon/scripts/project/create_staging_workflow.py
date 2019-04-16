#!/usr/bin/env python

from muon.database.database import Database
from muon.images.image_group import ImageGroup
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
    muon.config._instance.config['panoptes']['project_id'] = 1815

    pan.Uploader._client = pan.Panoptes.connect(
        endpoint='https://panoptes-staging.zooniverse.org')
    pan.Uploader._client.login()

    if not os.path.isdir(image_path):
        raise FileNotFoundError(image_path)


    ids = [9, 10, 11, 12, 13]

    database = Database(database_file)

    # new_group = image_storage.get_group(100)
    # with database.conn as conn:
        # database.ImageGroup.delete_group(conn, 100)
        # conn.commit()

    image_group = ImageGroup.load(0, database)
    image_group.images = {i: image_group.images[i] for i in ids}
    image_group.upload_subjects(image_path)

    import code
    code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()

