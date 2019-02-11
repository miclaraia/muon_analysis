
import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.database import Database
from muon.subjects.images import ImageStorage


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('image_dir')
def main(database_file, image_dir):
    database = Database(database_file)
    image_storage = ImageStorage(database)
    subject_storage = Storage(database)

    group = image_storage.get_group(0)
    group.generate_images(subject_storage, image_dir)


if __name__ == '__main__':
    main()
