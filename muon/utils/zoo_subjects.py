
from swap.db import DB
from muon.utils.camera import Camera, CameraPlot
import muon.utils.subjects

from collections import OrderedDict
import panoptes_client as pclient
import urllib.request
import os
import re
import numpy as np
import h5py
import random
import math
from skimage.io import imread
from sklearn import preprocessing
import progressbar
import matplotlib.pyplot as plt


class Subject(muon.utils.subjects.Subject):

    def __init__(self, subject, event, charge, score):
        super().__init__(subject, event, charge)
        self.label = score.label
        self.score = score.p

    def load_image(self):
        s = pclient.subject.Subject(self.id)
        url = list(s.raw['locations'][0].values())[0]
        return imread(url)

    def color(self):
        if self.label == -1:
            return (.8, .8, .8)
        elif self.label == 0:
            return (.1, .1, .8)
        return super().color()

    def __str__(self):
        return 'id %d event %s label %d score %f' % \
               (self.id, self.event, self.label, self.score)


class Subjects(muon.utils.subjects.Subjects):

    def load_images(self):
        for s in self.list():
            yield s.load_image()

    @staticmethod
    def get_swap_scores():
        return DB().subjects.get_scores()

    @classmethod
    def subjects_from_files(cls, paths):
        subjects = {}
        swap_scores = cls.get_swap_scores()
        for run, event, charge in cls.load_files(paths):
            evt = cls.parse_event(run, event)
            subject = cls.evt_to_subj(evt)

            if subject in swap_scores:
                charge = charge[:-1]
                score = swap_scores[subject]
                # if score.label in [0, 1]:
                s = Subject(subject, evt, charge, score)

                subjects[subject] = s

        return subjects
