
import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.database import Database
from muon.subjects.images import ImageStorage


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('image_dir')
@click.option('--groups', required=True)
def main(database_file, image_dir, groups):
    database = Database(database_file)
    image_storage = ImageStorage(database)
    subject_storage = Storage(database)

    for group in [int(g) for g in groups.split(',')]:
        print('Group', group)
        group = image_storage.get_group(group)
        for image in group.generate_images(subject_storage, image_dir):
            image_storage.update_image(image)


if __name__ == '__main__':
    main()
