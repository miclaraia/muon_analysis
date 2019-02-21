import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.database import Database
from muon.subjects.images import ImageStorage


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('cluster_name')
@click.option('--batches', required=True)
def main(database_file, cluster_name, batches):
    database = Database(database_file)
    image_storage = ImageStorage(database)
    subject_storage = Storage(database)

    kwargs = {
        'image_size': 36,
        'image_width': 6,
        'permutations': 1,
        'cluster_name': cluster_name
    }
    for batch in [int(b) for b in batches.split(',')]:
        print('batch', batch)
        image_storage.new_group(subject_storage, batch=batch, **kwargs)

    with image_storage.conn as conn:
        import code
        code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
