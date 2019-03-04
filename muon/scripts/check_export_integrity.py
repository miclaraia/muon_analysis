import csv
import json
import click
from tqdm import tqdm

from muon.subjects.database import Database
from muon.project.panoptes import Uploader


@click.group()
def cli():
    pass


@cli.command()
@click.argument('database_file')
@click.option('--group', type=int, required=True)
@click.option('--destructive', is_flag=True)
def online(database_file, group, destructive):
    database = Database(database_file)
    uploader = Uploader(5918, group)
    
    images = set()
    unlink = []
    set_zooid = 0
    with database.conn as conn:
        for subject in tqdm(uploader.get_subjects()):
            zoo_id = subject.id
            image_id = subject.metadata['id']

            image = database.Image.get_image(conn, image_id)
            print(zoo_id, image_id, image.zoo_id)

            if image_id in images:
                print('found duplicate image upload')
                unlink.append(zoo_id)
            else:
                images.add(image_id)

                if image.zoo_id != zoo_id:
                    set_zooid += 1
                    image.zoo_id = zoo_id
                    database.Image.update_zooid(conn, image)


        reset_zooid = 0
        print('Resetting image zooids')
        for image_id in tqdm(database.Image.get_group_images(conn, group)):
            if image_id not in images:
                image = database.Image.get_image(conn, image_id)
                if image.zoo_id != None:
                    reset_zooid += 1
                    image.zoo_id = None
                    database.Image.update_zooid(conn, image)

        print('Set {} reset {} unlinked {}'.format(
            set_zooid, reset_zooid, len(unlink)))

        if destructive:
            print('Unlinking subjects')
            uploader.unlink_subjects(unlink)
            print('Committing db changes')
            conn.commit()


@cli.command()
@click.argument('database_file')
@click.argument('subject_export')
@click.option('--group', type=int, required=True)
@click.option('--destructive', is_flag=True)
def from_export(database_file, subject_export, group, destructive):
    with open(subject_export, 'r') as file:
        existing_subjects = {}
        duplicate_subjects = []

        for row in csv.DictReader(file):
            image_id = json.loads(row['metadata'])['id']
            zoo_id = int(row['subject_id'])

            if image_id in existing_subjects:
                # Image was uploaded twice to zooniverse
                duplicate_subjects.append(zoo_id)
                print('image {} zooid {}'.format(image_id, zoo_id))
                raise Exception('Found duplicate image-subject mapping')
            else:
                existing_subjects[image_id] = zoo_id

    print('Found duplicate subject uploads')
    print(duplicate_subjects)
    if destructive:
        uploader = Uploader(5918, group)
        uploader.unlink_subjects(duplicate_subjects)

    database = Database(database_file)

    wrong_id = 0
    no_zoo = 0
    with database.conn as conn:
        for image_id in tqdm(database.Image.get_group_images(conn, group)):
            image = database.Image.get_image(conn, image_id)

            if image_id in existing_subjects:
                # this image is in the zooniverse
                zoo_id = existing_subjects[image_id]
                if image.zoo_id != zoo_id:
                    # this image does not have the right zoo_id
                    wrong_id += 1
                    print('wrong zoo id {}'.format(image_id))
                    image.zoo_id = zoo_id
                    database.Image.update_zooid(conn, image)

            else:
                if image.zoo_id is not None:
                    # This image is not on the zooniverse export
                    # but it does have an image_id
                    # probably tried to upload it and failed
                    no_zoo += 1
                    print('no zooniverse subject {} {}'.format(image_id, zoo_id))
                    image.zoo_id = None
                    database.Image.update_zooid(conn, image)

        conn.commit()

    print('wrong id {} no zoo subject {}'.format(wrong_id, no_zoo))


cli()
