import click
from tqdm import tqdm



@click.group(invoke_without_command=True)
@click.argument('save_dir')
@click.argument('database_file')
@click.option('--batches')
def main(save_dir, database_file, batches):
    from muon.subjects.storage import Storage
    from muon.subjects.database import Database
    from muon.project.clustering import Clustering

    from redec_keras.models.decv2 import Config
    database = Database(database_file)
    storage = Storage(database)

    config = Config.load(save_dir)
    config.type = 'decv2'

    if batches:
        batches = [int(b) for b in batches.split(',')]
    Clustering.assign_clusters(config, storage, config.name, batches=batches)


if __name__ == '__main__':
    main()
