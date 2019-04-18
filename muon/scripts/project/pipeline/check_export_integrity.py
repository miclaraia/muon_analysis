import csv
import json
import click
from tqdm import tqdm

from muon.database.database import Database
from muon.project.panoptes import Uploader
from muon.images.image_group import ImageGroup
from muon.images.image import Image
from muon.config import Config


@click.group()
def cli():
    pass


def check(zoo_subjects, group, database, uploader, destructive):
    images = set()
    unlink = []
    set_zooid = 0

    image_group = ImageGroup.load(group, database)
    image_group.images.load_all()
    for zoo_id, image_id in tqdm(zoo_subjects):
        # image = Image(image_id, database, online=destructive)
        image = image_group.images[image_id]
        print(zoo_id, image_id, image.zoo_id)

        if image_id in images:
            print('found duplicate image upload')
            unlink.append(zoo_id)
        else:
            images.add(image_id)

            if image.zoo_id != zoo_id:
                set_zooid += 1
                image.zoo_id = zoo_id


    reset_zooid = 0
    print('Resetting image zooids')
    # group = ImageGroup(group, database, online=destructive)
    for image in tqdm(image_group.images):
        if image.image_id not in images:
            if image.zoo_id != None:
                reset_zooid += 1
                image.zoo_id = None

    print('Set {} reset {} unlinked {}'.format(
        set_zooid, reset_zooid, len(unlink)))

    if destructive:
        print('Unlinking subjects')
        uploader.unlink_subjects(unlink)


@cli.command()
@click.option('--config')
@click.option('--group', type=int, required=True)
@click.option('--destructive', is_flag=True)
def online(config, group, destructive):
    Config.new(config)
    database = Database()
    uploader = Uploader(5918, group)

    def zoo_subjects():
        for subject in uploader.get_subjects():
            yield subject.id, subject.metadata['id']

    check(zoo_subjects(), group, database, uploader, destructive)


@cli.command()
@click.option('--config')
@click.argument('subject_export')
@click.option('--group', type=int, required=True)
@click.option('--destructive', is_flag=True)
def from_export(config, subject_export, group, destructive):
    Config.new(config)
    database = Database()
    uploader = Uploader(5918, group)

    def zoo_subjects():
        with open(subject_export, 'r') as f:
            for row in csv.DictReader(f):
                zoo_id = int(row['subject_id'])
                metadata = json.loads(row['metadata'])
                image_id = metadata['id']
                group_id = metadata['#group']

                if group_id == group:
                    yield zoo_id, image_id

    check(zoo_subjects(), group, database, uploader, destructive)


cli()
