import click
from tqdm import tqdm
import logging

from muon.subjects.storage import Storage
from muon.database.database import Database
from muon.images.image_group_single import SingleImageGroup

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.option('--batches', required=True)
def main(database_file, batches):
    database = Database(database_file)

    with database.conn as conn:

        for batch in [int(b) for b in batches.split(',')]:

            subject_ids = database.Subject \
                .get_subject_ids_in_batch(conn, batch)
            print('batch', batch)
            subject_ids = list(subject_ids)
            print(subject_ids)

            import code
            subject_storage = Storage(database)
            code.interact(local={**globals(), **locals()})

            description = 'Batch {} single images'.format(batch)
            group = SingleImageGroup.new(
                database, subject_ids, description=description)
    print(group)

    with database.conn as conn:
        import code
        subject_storage = Storage(database)
        code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
