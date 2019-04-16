import click
from tqdm import tqdm
import random
import os

from muon.subjects.storage import Storage
from muon.database.database import Database
from muon.images.image_group import ImageGroup
from muon.images.image_group_single import SingleImageGroup


database_file = os.path.join(os.getenv('MUOND'), 'subjects-main/data-main.db')
image_dir = os.path.join(os.getenv('MUOND'), 'images/mh2_images-main')


def main():
    N = 25000
    batch = 13

    database = Database(database_file)
    with database.conn as conn:


        subject_ids = database.Subject \
            .get_subject_ids_in_batch(conn, batch)

        subject_ids = list(subject_ids)
        subject_ids = random.sample(subject_ids, N)
        print(subject_ids)

        description = 'Batch {} single images'.format(batch)
        group = SingleImageGroup.new(
            database, subject_ids, description=description)
    print(group)

    subject_storage = Storage(database)
    group.generate_images(subject_storage, path=image_dir)


main()
