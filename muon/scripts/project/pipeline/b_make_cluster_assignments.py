import click
from tqdm import tqdm



@click.group(invoke_without_command=True)
@click.argument('save_dir')
@click.option('--config')
@click.option('--batches')
def main(save_dir, config, batches):
    from muon.subjects.storage import Storage
    from muon.database.database import Database
    from muon.project.clustering import Clustering
    import muon.config

    from redec_keras.models.decv2 import Config

    muon.config.Config.new(config)
    database = Database()
    storage = Storage(database)

    config = Config.load(save_dir)
    config.type = 'decv2'

    if batches:
        batches = [int(b) for b in batches.split(',')]
    Clustering.assign_clusters(config, storage, config.name, batches=batches)


if __name__ == '__main__':
    main()
