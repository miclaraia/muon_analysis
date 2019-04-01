import click

from muon.database.database import Database
from muon.project.parse_export import Aggregate, load_config


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('config_file')
@click.argument('classification_export')
def main(database_file, config_file, classification_export):
    database = Database(database_file)
    config = load_config(config_file)

    agg = Aggregate(config, database)

    agg.aggregate(classification_export)

    import code
    code.interact(local={**globals(), **locals()})


main()
