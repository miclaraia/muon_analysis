import click
import numpy as np
import os
import json
import random

from muon.subjects.database import Database
from muon.subjects.images import ImageStorage, Image, ImageGroup
from muon.subjects.storage import Storage
import muon.project.panoptes as pan
import muon.config


@click.group()
def cli():
    pass

@cli.command()
@click.argument('database_file')
def find_muons(database_file):
    database = Database(database_file)
    subject_storage = Storage(database)

    with database.conn as conn:
        group_id = database.ImageGroup.next_id(conn)
        image_id = database.Image.next_id(conn)

        print('getting subjects')
        cursor = conn.execute("""
            SELECT subject_id FROM subject_labels
            WHERE label_name=? AND label=?
            LIMIT 200""", ('vegas', 1))
        # cursor = conn.execute("""
            # SELECT subject_id FROM subject_labels
            # INNER JOIN subjects
                # ON subject_labels.subject_id=subjects.subject_id
            # WHERE subjects.source=? AND label_name=? AND label=?
            # LIMIT 1000""", ('vegas', 1))

        muons = [row[0] for row in cursor]
        print(muons)

    image_path = '/tmp/muon-tutorial-image'

    print('getting subjects')
    subjects = subject_storage.get_subjects(muons)
    fig = subjects.plot_subjects(w=5, grid=True)
    fig.savefig(image_path+'/test.png')

    with open(image_path+'/test_subjects.json', 'w') as f:
        json.dump(list(subjects.keys()), f)


@cli.command()
@click.argument('database_file')
@click.argument('image_path')
def gen_100(database_file, image_path):
    database = Database(database_file)
    subject_storage = Storage(database)

    with database.conn as conn:
        print('getting subjects')
        cursor = conn.execute("""
            SELECT subject_id FROM subject_labels
            WHERE label_name=? AND label=?
            LIMIT 100""", ('vegas', 1))

        muons = [row[0] for row in cursor]
        print(muons)

    print('getting subjects')

    subjects = subject_storage.get_subjects(muons)
    fig = subjects.plot_subjects(w=10, grid=True)
    fig.savefig(image_path)


@cli.command()
@click.argument('database_file')
def generate(database_file):
    image_path = '/tmp/muon-tutorial-image'
    with open(image_path+'/test_subjects.json', 'r') as f:
        subjects = json.load(f)

    indeces = """1-1 , 2-1 , 2-4 , 4-3 , 5-1 ,
                 6-3 , 6-4 , 7-5 , 10-1, 12-2,
                 12-3, 12-5, 14-1, 14-5, 15-3,
                 16-2, 17-4, 20-3, 20-4, 21-4, 
                 24-2, 25-5, 26-2, 26-4, 27-4,
                 28-2, 28-3, 29-4, 30-1, 30-3,
                 1-2 , 1-3 , 1-4 , 1-5 , 2-2, 3-1
            """

    i2 = []
    for i in indeces.split(','):
        a, b = i.strip().split('-')
        a = int(a)
        b = int(b)
        i2.append((a-1)*5+b-1)
    subjects = [subjects[i] for i in i2]
    random.shuffle(subjects)

    database = Database(database_file)
    subject_storage = Storage(database)

    with database.conn as conn:
        image_id = database.Image.next_id(conn)
        group_id = database.ImageGroup.next_id(conn)

    image = Image(image_id, group_id, None, subjects, {})
    print(image)
    group = ImageGroup(group_id, '', {image_id: image})
    print(group)

    image_path = '/tmp/muon-tutorial-image'
    if not os.path.isdir(image_path):
        os.mkdir(image_path)

    print('Generating images')
    list(group.generate_images(subject_storage, dpi=100, path=image_path))

    muon.config.project = 1815
    pan.Uploader._client = pan.Panoptes.connect(
        endpoint='https://panoptes-staging.zooniverse.org')
    pan.Uploader._client.login()

    list(group.upload_subjects(image_path))


if __name__ == '__main__':
    cli()


