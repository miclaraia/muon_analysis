import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.database import Database
from muon.subjects.images import ImageStorage


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('cluster_name')
@click.option('--batch', nargs=1, type=int)
def main(database_file, cluster_name, batch):
    database = Database(database_file)
    image_storage = ImageStorage(database)
    subject_storage = Storage(database)

    kwargs = {
        'image_size': 36,
        'image_width': 6,
        'permutations': 1,
        'batch': batch,
        'cluster_name': cluster_name
    }
    image_storage.new_group(subject_storage, **kwargs)

    with image_storage.conn as conn:
        import code
        code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
