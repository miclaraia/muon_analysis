import click
from tqdm import tqdm
import logging

from muon.subjects.storage import Storage
from muon.subjects.database import Database
from muon.subjects.images import ImageGroup

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
        group_id = database.ImageGroup.next_id(conn)

        for batch in [int(b) for b in batches.split(',')]:
            print('batch', batch)

            cluster_assignments = database.Clustering \
                .get_cluster_assignments(conn, cluster_name, batch)

            if not cluster_assignments:
                logger.warn('cluster_assignments: {}'.format(cluster_assignments))
                raise Exception('Empty cluster assignment struct')

    group = ImageGroup.new(group_id, database, cluster_name,
                           cluster_assignments, **kwargs)
    print(group)

    with database.conn as conn:
        import code
        code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
