
from muon.ui import ui
from muon.utils.subjects import Subjects, Subject_Data
from muon.utils.camera import Camera, CameraPlot

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


# def load_subjects(path):
    # print(path)
    # print(os.path.splitext(path[0]))
    # if len(path) == 1 and os.path.splitext(path[0])[1] == '.pkl':
        # subjects = pickle.load(open(path[0], 'rb'))
    # else:
        # subjects = Subjects.from_data(path)
    # return subjects


def interact(local):
    code.interact(local={**globals(), **locals(), **local})

# @subjects.command()
# @click.argument('path', nargs=-1)
# def plot(path):
    # subjects = load_subjects(path)

    # fig = plt.figure()
    # s = subjects._sample_s(20).plot_subjects(fig, 5)

    # interact(locals())

@subjects.command()
@click.argument('fname', nargs=1)
@click.argument('data', nargs=1)
def save(fname, data):
    subjects = Subjects.from_data(data)
    pickle.dump(subjects, open(fname, 'wb'))


@subjects.command()
@click.argument('output', nargs=1)
@click.argument('raw', nargs=-1)
def generate(output, raw):
    sd = Subject_Data(output)
    sd.load_raw(raw)
    sd.close()


@subjects.command()
def test():
    from muon.utils.subjects import Subject_Data
    sd = Subject_Data('test.hdf5')
    sd.load_raw(['hdf5/78573muon_hunter_events_oversampled.hdf5'])

    interact(locals())
