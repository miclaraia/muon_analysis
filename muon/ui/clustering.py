from muon.ui import ui
from muon.utils.clustering import Cluster
from muon.utils.subjects import Subjects
import muon.scripts.clustering as scripts
import swap.config

import click
import code
import pickle
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


@ui.cli.group()
def pca():
    pass


def interact(local):
    def save(path):
        pickle.dump(local['cluster'], open(path, 'wb'))

    code.interact(local={**globals(), **locals(), **local})


@pca.command()
@click.argument('path', nargs=-1)
@click.option('--components', default=8, type=int)
@click.option('--plot/--noplot', default=True)
@click.option('--mle', is_flag=True)
@click.option('--all', is_flag=True)
@click.option('--save', nargs=1, type=str)
def run(path, components, mle, plot, save, all):
    if len(path) == 0:
        print('No data files provided')
        return

    swap.config.logger.init()
    subjects = Subjects.from_data(path)
    logger.info('Done loading subjects')

    if mle:
        cluster = Cluster.create(subjects, 'mle')
    if all:
        cluster = Cluster.create(subjects, None)
    else:
        cluster = Cluster.create(subjects, components)
    logger.info('Done creating cluster')

    if plot:
        try:
            cluster.plot()
        except Exception:
            pass

    if save:
        pickle.dump(cluster, open(save, 'wb'))

    interact(locals())

@pca.command()
@click.argument('path', nargs=1)
def load(path):
    cluster = pickle.load(open(path, 'rb'))

    plot = lambda: scripts.plot_classes(cluster)
    interact(locals())

@pca.command()
@click.argument('path', nargs=1)
def visualize(path):
    swap.config.logger.init()
    subjects = Subjects.from_data(path)
    cluster = Cluster.create(subjects)
    cluster.visualize()


@pca.command()
@click.argument('cluster', nargs=1)
@click.argument('path', nargs=1)
def region(cluster, path):
    cluster = pickle.load(open(cluster, 'rb'))
    scripts.regions(cluster, path)
