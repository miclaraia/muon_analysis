import click
from tqdm import tqdm
import logging

from muon.subjects.storage import Storage
from muon.database.database import Database
from muon.images.image_group import ImageGroup

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('cluster_name')
@click.option('--batches', required=True)
def main(database_file, cluster_name, batches):
    database = Database(database_file)
    subject_storage = Storage(database)

    kwargs = {
        'image_size': 36,
        'image_width': 6,
        'permutations': 1,
    }

    with database.conn as conn:

        for batch in [int(b) for b in batches.split(',')]:
            print('batch', batch)

            cluster_assignments = database.Clustering \
                .get_cluster_assignments(conn, cluster_name, batch)

            if not cluster_assignments:
                logger.warn('cluster_assignments: {}'.format(cluster_assignments))
                raise Exception('Empty cluster assignment struct')

            description = 'Batch {}'.format(batch)

            group = ImageGroup.new(database, cluster_name,
                                   cluster_assignments,
                                   description=description,
                                   **kwargs)
    print(group)

    with database.conn as conn:
        import code
        code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
