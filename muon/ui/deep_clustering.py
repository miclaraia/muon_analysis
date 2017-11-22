from muon.ui import ui
from muon.utils.subjects import Subjects
from muon.deep_clustering.clustering import Config, Cluster
import swap.config

import click
import code
import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

@ui.cli.group()
def dec():
    pass


def interact(local):
    def save(path):
        pickle.dump(local['cluster'], open(path, 'wb'))

    code.interact(local={**globals(), **locals(), **local})


@dec.command()
@click.argument('output', nargs=1)
@click.argument('weights', nargs=1)
@click.argument('path', nargs=-1)
@click.option('--save', nargs=1, type=str)
def run(output, weights, path, save):
    if len(path) == 0:
        print('No data files provided')
        return

    swap.config.logger.init()
    subjects = Subjects.from_data(path)
    logger.info('Done loading subjects')

    config = Config(**{
        'n_clusters': 10,
        'batch_size': 256,
        'lr': 0.01,
        'momentum': 0.9,
        'tol': 0.001,
        'maxiter': 2e4,
        'update_interval': 140,
        'save_dir':  output,
        'ae_weights': weights,
    })

    cluster = Cluster.create(subjects, config)
    logger.info('Done training network')

    # if save:
        # pickle.dump(cluster, open(save, 'wb'))

    interact(locals())
