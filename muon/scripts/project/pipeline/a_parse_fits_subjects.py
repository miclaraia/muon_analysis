import click
from tqdm import tqdm
import os
import logging
import csv

from muon.subjects.storage import Storage
from muon.subjects.subjects import Subject
from muon.subjects.parsing import ParseFits
from muon.database.database import Database
from muon.subjects.source import Source
from muon.config import Config

logger = logging.getLogger(__name__)


def add_file(input_file, storage, database, batch):

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
                source.hash == None
                source.save()
        else:
            logger.info('Creating new source')
            source = Source.new(source_id, database, location)

    parser = ParseFits.parse_file(input_file)

    if batch:
        storage.add_subjects(parser, batch, 'vegas')
    else:
        storage.add_batch(parser, 'vegas')

    source.update_hash(location)
    source.save()


@click.group()
def cli():
    pass


@cli.command()
@click.argument('manifest')
@click.option('--config')
@click.option('--batch', nargs=1, type=int)
def one(input_file, config, batch):
    Config.new(config)
    database = Database()
    storage = Storage(database)

    add_file(input_file, storage, database, batch)


@cli.command()
@click.argument('manifest')
@click.option('--config')
def manifest(manifest, config):
    Config.new(config)
    database = Database()
    storage = Storage(database)

    path = os.path.dirname(manifest)

    with open(manifest, 'r') as file:
        for row in csv.DictReader(file):
            if int(row['active']):
                batch = int(row['batch'])
                if batch == -1:
                    continue

                fname = os.path.join(path, row['name'])
                print(batch, fname)

                add_file(fname, storage, database, batch)



if __name__ == '__main__':
    cli()
