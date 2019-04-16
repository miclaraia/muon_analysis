
import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.database.database import Database
from muon.images.image_group import ImageGroup


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('image_dir')
@click.option('--groups', required=True)
@click.option('--dpi', type=int)
def main(database_file, image_dir, groups, dpi):
    database = Database(database_file)
    subject_storage = Storage(database)

    for group in [int(g) for g in groups.split(',')]:
        print('Group', group)
        group = ImageGroup.load(group, database, online=True)
        print(group)
        group.generate_images(
            subject_storage, path=image_dir, dpi=dpi)


if __name__ == '__main__':
    main()