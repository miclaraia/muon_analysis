
from muon.ui import ui
from muon.utils.subjects import Subjects
from muon.deep_clustering.clustering import Config, Cluster
from muon.utils.camera import Camera, CameraPlot
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
def subjects():
    pass


def load_subjects(path):
    print(path)
    print(os.path.splitext(path[0]))
    if len(path) == 1 and os.path.splitext(path[0])[1] == '.pkl':
        subjects = pickle.load(open(path[0], 'rb'))
    else:
        subjects = Subjects.from_data(path)
    return subjects


def interact(local):
    code.interact(local={**globals(), **locals(), **local})

@subjects.command()
@click.argument('path', nargs=-1)
def plot(path):
    subjects = load_subjects(path)

    fig = plt.figure()
    s = subjects._sample_s(20).plot_subjects(fig, 5)
    # fig.show()

    interact(locals())

@subjects.command()
@click.argument('fname', nargs=1)
@click.argument('path', nargs=-1)
def save(fname, path):
    subjects = load_subjects(path)
    pickle.dump(subjects, open(fname, 'wb'))


@subjects.command()
def test():
    data = [1 for x in range(499)]
    data = Camera().transform(data, False)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, xticks=[], yticks=[])
    CameraPlot.plot(data, ax)
    # fig.show()

    interact(locals())
