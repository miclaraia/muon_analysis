
from muon.ui import ui
from muon.utils.subjects import Subjects
from muon.deep_clustering.supervised import Config, Supervised
import muon.project.parse_export as pe

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
def supervised():
    pass


def interact(local):
    def save(path):
        pickle.dump(local['cluster'], open(path, 'wb'))

    code.interact(local={**globals(), **locals(), **local})


def load_subjects(fname):
    with open(fname, 'rb') as file:
        subjects = pickle.load(file)
    logger.info('Done loading subjects')

    agg = pe.Aggregate.load('mh2_gold')
    _s = list(agg.data['subjects'].keys())
    subjects = subjects.subset(_s)
    agg.apply_labels(subjects)
    return subjects



@supervised.command()
@click.argument('output', nargs=1)
@click.argument('subjects', nargs=1)
@click.option('--ae-weights', nargs=1)
def run(output, subjects, ae_weights):

    subjects_ = load_subjects(subjects)

    config = Config(**{
        'input_shape': subjects_.dimensions,
        'loss': 'categorical_crossentropy',
        'optimizer': 'adam',
        'rotation': True,
        'save_dir': output,
        'ae_weights': ae_weights,
        'subjects': subjects,
    })
    logger.info(config)
    config.dump()


    model = Supervised(config)
    logger.info('Training model')
    model.train(subjects_)

    model.save()
    code.interact(local={**globals(), **locals()})


@supervised.command()
@click.argument('config', nargs=1)
def load(config):
    config = Config.load(config)
    subjects = load_subjects(config.subjects)
    model = Supervised.load(config)

    logger.info('scores with rotation: %s', model.score(subjects))
    config.rotation = False
    logger.info('scores without rotation: %s', model.score(subjects))

    interact(locals())

