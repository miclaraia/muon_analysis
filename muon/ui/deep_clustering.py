from muon.ui import ui
from muon.utils.subjects import Subjects
from muon.deep_clustering.clustering import Config, Cluster
import swap.config

import os
import click
import code
import pickle
import numpy as np
import random
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


def load_subjects(path):
    print(path)
    print(os.path.splitext(path[0]))
    if len(path) == 1 and os.path.splitext(path[0])[1] == '.pkl':
        subjects = pickle.load(open(path[0], 'rb'))
        path = path[0]
    else:
        subjects = Subjects.from_data(path)
        path = None
    return subjects, path


@dec.command()
@click.argument('output', nargs=1)
@click.argument('path', nargs=-1)
@click.option('--ae-weights', nargs=1)
@click.option('--clusters', nargs=1)
def run(output, path, ae_weights, clusters):
    if len(path) == 0:
        print('No data files provided')
        return

    swap.config.logger.init()
    # subjects = Subjects.from_data(path)
    subjects, fname = load_subjects(path)

    if fname is None:
        fname = os.path.join(output, 'subjects.pkl')
        logger.info('saving subjects to %s', fname)
        pickle.dump(subjects, open(fname, 'wb'))

    logger.info('Done loading subjects')

    config = Config(**{
        'n_clusters': int(clusters or 10),
        'batch_size': 256,
        'lr': 0.01,
        'momentum': 0.9,
        'tol': 0.001,
        'maxiter': 2e4,
        'update_interval': 140,
        'save_dir':  output,
        'ae_weights': ae_weights or os.path.join(output, 'ae_weights.h5'),
        'subjects': fname,
    })
    config.dump()

    cluster = Cluster.create(subjects, config)

    logger.info('Training model')
    cluster.train()
    pred = cluster.predictions
    logger.info('Done training network')

    # if save:
        # pickle.dump(cluster, open(save, 'wb'))

    interact(locals())

@dec.command()
@click.argument('config', nargs=1)
def load(config):
    config = Config.load(config)
    subjects = pickle.load(open(config.subjects, 'rb'))
    cluster = Cluster.create(subjects, config)

    logger.info('Training model')
    cluster.train()
    pred = cluster.predictions
    logger.info('Done training network')

    interact(locals())

