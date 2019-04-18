
import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.database.database import Database
from muon.images.image_group import ImageGroup
from muon.config import Config


@click.group(invoke_without_command=True)
@click.option('--config')
@click.option('--groups', required=True)
@click.option('--dpi', type=int)
def main(config, groups, dpi):
    Config.new(config)
    database = Database()
    subject_storage = Storage(database)

    for group in [int(g) for g in groups.split(',')]:
        print('Group', group)
        group = ImageGroup.load(group, database, online=True)
        print(group)
        group.generate_images(subject_storage, dpi=dpi)


if __name__ == '__main__':
    main()
