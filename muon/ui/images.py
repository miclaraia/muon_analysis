
from muon.ui import ui
from muon.deep_clustering.clustering import Config, Cluster
from muon.project.images import Images, Random_Images
import muon.project.panoptes as panoptes
import muon.config

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
@click.argument('subjects', nargs=1)
def test(subjects):
    subjects = pickle.load(open(subjects, 'rb'))
    images = Images.new(subjects)
    interact(locals())


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
    images.save_group(overwrite=True)

    interact(locals())


@images.command()
@click.argument('group', type=int)
@click.option('--config', nargs=1)
@click.option('--structure', nargs=1)
def load(group, config, structure):
    if config:
        config = Config.load(config)
        subjects = pickle.load(open(config.subjects, 'rb'))

    if structure:
        images = Random_Images.load_group(group, fname=structure)
    else:
        images = Random_Images.load_group(group)
    print(images)
    interact(locals())


@images.command()
@click.argument('group', type=int)
@click.argument('path')
def upload(group, path):
    images = Images.load_group(group)
    images.upload_subjects(path)

    interact(locals())


@images.command()
@click.argument('group', type=int)
def unlink(group):
    print('Unlinking subjects')
    images = Images.load_group(group)
    to_remove = []
    for i in images.iter():
        if i.zoo_id is not None:
            to_remove.append(i.zoo_id)
            i.zoo_id = None
    print('Unlinking %d subjects' % len(to_remove))
    if len(to_remove) > 0:
        uploader = panoptes.Uploader(muon.config.project, group)
        uploader.unlink_subjects(to_remove)
        images.save_group(overwrite=True)


@images.command()
def list():
    groups = Images._list_groups()
    print('%d groups in images file:' % len(groups))
    print(' '.join(groups))



