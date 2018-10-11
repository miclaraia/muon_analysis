

import os
import csv
import click
import code
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import logging

from muon.ui import ui
from muon.subjects import Subjects
from muon.subjects.storage import Storage
from muon.utils.camera import Camera, CameraPlot, CameraRotate

logger = logging.getLogger(__name__)


@ui.cli.group()
def subjects():
    pass


@subjects.command()
@click.argument('name')
@click.argument('in_file')
@click.argument('subjects_data')
def add_labels(name, labels_csv, subjects_data):
    storage = Storage(subjects_data)

    def label_generator():
        with open(labels_csv, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                yield row['subject'], row['label']

    storage.add_labels(name, label_generator())
