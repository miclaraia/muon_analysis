#!/usr/bin/env python
import click
import os
import logging
from tqdm import tqdm

from muon.subjects.database import Database
from muon.subjects.images import ImageStorage, ImageGroup
import muon.project.panoptes as pan

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(logger.level, logging.DEBUG)
logging.warn('%s', logger.level)

@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('image_path')
@click.option('--groups', required=True)
def main(database_file, image_path, groups):
    database = Database(database_file)
    image_storage = ImageStorage(database)
    pan.Uploader.client()

    for group in [int(g) for g in groups.split(',')]:
        logger.info('Group %d', group)
        image_group = image_storage.get_group(group)

        for image in tqdm(image_group.upload_subjects(image_path)):
            logger.info('image %s', str(image))
            image_storage.update_image_zooid(image)

if __name__ == '__main__':
    main()
