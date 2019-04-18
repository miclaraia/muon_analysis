#!/usr/bin/env python
"""
Make test/train split assignment
Create a model config
send it to aws? or assume it's running on aws
    nah assume its running on aws
"""

import click
from tqdm import tqdm
import os

from muon.subjects.storage import Storage
from muon.database.database import Database
from muon.project.clustering import Clustering
import muon.config

from redec_keras.models.decv2 import Config


@click.group(invoke_without_command=True)
@click.option('--name')
@click.option('--save_dir')
@click.option('--config')
@click.option('--batches')
def main(name, save_dir, config, batches):
    muon.config.Config.new(config)
    database = Database()
    storage = Storage(database)

    # Clustering.reserve_test_set(storage)

    config_args = {
        'save_dir': save_dir,
        'name': name,
        'source_dir': None,
        'splits_file': None,
        'n_classes': 2,
        'n_clusters': 50,
        'source_weights': (None, None),
    }

    if os.path.isdir(save_dir):
        print('save_dir', save_dir)
        raise FileExistsError('save_dir already exists!')
    os.mkdir(save_dir)
    config = Config(**config_args)
    config.dump()

    if batches:
        batches = [int(b) for b in batches.split(',')]
    print(batches)
    subjects = storage.get_split_subjects('train', batches=batches)
    x = subjects.get_x()
    del subjects

    Clustering.train_decv2(config, x)


if __name__ == '__main__':
    main()
