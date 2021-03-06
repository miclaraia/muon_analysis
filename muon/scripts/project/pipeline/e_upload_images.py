#!/usr/bin/env python
import click
import os
import logging
from tqdm import tqdm
import csv
import json

from muon.database.database import Database
from muon.images.image_group import ImageGroup
import muon.project.panoptes as pan
from muon.config import Config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(logger.level, logging.DEBUG)
logging.warn('%s', logger.level)

@click.group(invoke_without_command=True)
@click.option('--config')
@click.option('--subject_export')
@click.option('--groups', required=True)
def main(config, subject_export, groups):
    Config.new(config)
    database = Database()
    pan.Uploader.client()

    if subject_export:
        logger.info('Loading existing subject export')
        with open(subject_export, 'r') as file:
            existing_subjects = {}
            for row in csv.DictReader(file):
                image_id = json.loads(row['metadata'])['id']
                zoo_id = int(row['subject_id'])

                if image_id in existing_subjects:
                    print('image {} zooid {}'.format(image_id, zoo_id))
                    raise Exception('Found duplicate image-subject mapping')
                else:
                    existing_subjects[image_id] = zoo_id
    else:
        existing_subjects = None

    for group_id in [int(g) for g in groups.split(',')]:
        logger.info('Group %d', group_id)
        image_group = ImageGroup.load(group_id, database, online=True)

        image_group.upload_subjects(existing_subjects=existing_subjects)


if __name__ == '__main__':
    main()
