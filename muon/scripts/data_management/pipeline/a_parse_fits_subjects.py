import click
from tqdm import tqdm
import os
import logging

from muon.subjects.storage import Storage
from muon.subjects.subjects import Subject
from muon.subjects.parsing import ParseFits
from muon.database.database import Database
from muon.subjects.source import Source

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.argument('input_file')
@click.argument('database_file')
@click.option('--batch', nargs=1, type=int)
def main(input_file, database_file, batch):
    database = Database(database_file)
    storage = Storage(database)

    parser = ParseFits.parse_file(input_file)

    subjects = []
    def wrapper():
        for subject in parser:
            subjects.append(subject)
            yield subject

    location, source_id = os.path.split(input_file)
    with database.conn as conn:
        logger.info('Updating source hash')
        if database.Source.source_exists(conn, input_file):
            source = Source(input_file, database)
            logger.info('original hash: %s', source.hash)

            if source.compare(location):
                logger.info('Nothing to do, hashes match')
                return
            else:
                source.update_hash(location)
        else:
            logger.info('Creating new source')
            source = Source.new(source_id, database, location)

    if batch:
        storage.add_subjects(wrapper(), batch)
    else:
        storage.add_batch(wrapper())

    subject_labels = [(subject.id, subject.y) for subject in subjects]
    storage.add_labels('vegas', subject_labels)


if __name__ == '__main__':
    main()
