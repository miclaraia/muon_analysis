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
# database_file = os.path.join(os.getenv('MUOND'), 'subjects/ralph_subjects/data.db')
# image_dir = os.path.join(os.getenv('MUOND'), 'images')


def main():
    N = 25000
    batch = 13

    database = Database(database_file)
    with database.conn as conn:
        groups = []

        query = """
            SELECT subject_id FROM image_subjects
            WHERE group_id=13
            ORDER BY RANDOM()
            LIMIT 5000;
        """
        cursor = conn.execute(query)
        subject_ids = [row[0] for row in cursor]

        description = 'Single images (sims)'
        group = SingleImageGroup.new(
            database, subject_ids, description=description)
        groups.append(group)

        query = """
            SELECT subject_id FROM image_subjects
            WHERE group_id IN (10,11,12)
            ORDER BY RANDOM()
            LIMIT 5000;
        """
        cursor = conn.execute(query)
        subject_ids = [row[0] for row in cursor]

        description = 'Single images (real)'
        group = SingleImageGroup.new(
            database, subject_ids, description=description)
        groups.append(group)

    print(groups)
    input('Continue?')
    subject_storage = Storage(database)
    for group in groups:
        print(group)
        group.generate_images(subject_storage, path=image_dir)

main()
