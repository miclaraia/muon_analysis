
from muon.ui import ui
from muon.deep_clustering.clustering import Config, Cluster
from muon.project.images import Random_Images
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
def images():
    pass


def interact(local):
    def save(path):
        pickle.dump(local['cluster'], open(path, 'wb'))

    code.interact(local={**globals(), **locals(), **local})


@images.command()
@click.argument('config', nargs=1)
@click.option('--size', type=int)
@click.option('--width', type=int)
@click.option('--permutations', type=int)
@click.option('--save', is_flag=True)
def new(config, width, size, permutations, save):
    config = Config.load(config)
    subjects = pickle.load(open(config.subjects, 'rb'))
    cluster = Cluster.create(subjects, config)

    logger.info('Training model')
    cluster.train()
    logger.info('Done training network')

    kwargs = {}
    if width:
        kwargs['width'] = width
    if size:
        kwargs['image_size'] = size
    if permutations:
        kwargs['permutations'] = permutations

    images = Random_Images.new(cluster, **kwargs)

    if save:
        images.save_group()

    interact(locals())


@images.command()
@click.argument('config', nargs=1)
@click.argument('group', type=int)
@click.option('--path')
def generate(config, group, path):
    config = Config.load(config)
    subjects = pickle.load(open(config.subjects, 'rb'))

    images = Random_Images.load_group(group)
    images.generate_images(subjects, path)

    interact(locals())
