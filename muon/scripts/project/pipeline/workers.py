import click

from muon.images.workers import Workers
from muon.config import Config
from muon.database.database import Database


@click.group()
@click.option('--config')
def cli(config):
    if config:
        Config.new(config)


@cli.command()
@click.option('--group', required=True, type=int)
def generate(group):
    config = Config.instance()
    database = Database()
    worker = Workers(database)

    worker.generate_images(group)


@cli.command()
@click.option('--group', required=True, type=int)
def upload(group):
    config = Config.instance()
    database = Database()
    worker = Workers(database)

    worker.upload_images(group)


@cli.command()
def run():
    database = Database()
    worker = Workers(database)
    worker.run()


cli()
