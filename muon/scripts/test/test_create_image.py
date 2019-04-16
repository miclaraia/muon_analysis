import click
import code

from muon.subjects.database import Database
from muon.subjects.images import Image


@click.group()
def cli():
    pass


@cli.command
@click.argument('database_file')
def create(database_file):
    database = Database(database_file)
    with database.conn as conn:
        subjects = database.Subject.list_subjects(conn)[:36]
        image_id = database.Image.next_id(conn)
        group_id = database.ImageGroup.next_id(conn)

    image = Image.new(image_id, database, group_id, 0, {}, subjects)

@cli.command()
@click.argument('database_file')
def load(database_file):
    database = Database(database_file)
    image = Image(2000, database)

    code.interact(local={**globals(), **locals()})

cli()
