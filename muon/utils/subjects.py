
from swap.db import DB
from muon.utils.camera import Camera, CameraPlot

from collections import OrderedDict
import os
import re
import numpy as np
import h5py
import random
import math
from sklearn import preprocessing
import progressbar
import matplotlib.pyplot as plt


class Subject:

    def __init__(self, subject, event, charge):
        self.id = subject
        self.event = event
        self.charge = np.array(charge)
        self.scaled_charge = None

        self._normalize()

    def plot(self, ax, camera=None):
        if camera is None:
            camera = Camera()

        # x, y, c = camera.transform(self.charge)
        # ax.scatter(x, y, c=c, s=10, cmap='viridis')

        data = camera.transform(self.charge, False)
        CameraPlot.plot(data, ax, radius=camera.pixSideLength)
        return ax

    @staticmethod
    def color():
        return (.9, .1, .1)

    def _normalize(self):
        n = np.linalg.norm(self.charge)
        d = self.charge.shape[0]
        if n == 0:
            self.scaled_charge = self.charge.copy()
        else:
            c = self.charge / n * math.sqrt(d)
            self.scaled_charge = c

    def __str__(self):
        return 'id %d event %s' % \
               (self.id, self.event)

class Subjects:

    _mapping = None
    patterns = {
        'run': re.compile('run([0-9]+)'),
        'evt': re.compile('evt([0-9]+)'),
        'tel': re.compile('tel([0-9]+)'),
    }

    def __init__(self, subjects):
        if type(subjects) is list:
            subjects = [(s.id, s) for s in subjects]
        self.subjects = OrderedDict(subjects)

        self.dimensions = self._dimensions(self.list())

    @classmethod
    def from_data(cls, path, subject_mapping):
        # TODO write a script that makes the subject_mapping!
        subject_data = cls.subjects_from_files(path, subject_mapping)

        subjects = cls(subject_data)
        # subjects.scale_charges()

        return subjects

    def sample(self, size):
        size = int(size)
        subjects = list(self.subjects.values())
        print('number of subjects', len(subjects))
        if size > len(subjects):
            return subjects
        return random.sample(subjects, size)

    ##########################################################################
    ###   Subsets   ##########################################################
    ##########################################################################

    def _sample_s(self, size):
        return self.__class__(self.sample(size))

    def list(self):
        return list(self.subjects.values())

    def subject_ids(self):
        return list(self.subjects.keys())

    def subset(self, subjects):
        subset = [self.subjects[s] for s in subjects]
        return self.__class__(subset)

    # def labels(self, order):
        # labels = np.zeros(len(order))
        # for i, s in enumerate(order):
            # l = self.subjects[s].label
            # if l == -1:
                # l = None
            # labels[i] = l

        # return labels

    def labeled_subjects(self):
        subjects = []
        for s in self.list():
            if s.label in [0, 1]:
                subjects.append(s)

        return self.__class__(subjects)

    def sorted_subjects(self, method='swap'):
        if method == 'swap':
            s = sorted(self.list(), key=lambda s: s.score)
        return self.__class__(s)

    @staticmethod
    def _dimensions(subjects):
        l = len(subjects)
        if l > 0:
            w = len(subjects[0].charge)
            return l, w
        return l,

    @classmethod
    def evt_to_subj(cls, evt, mapping):
        """
        Get subject associated with specific run event and telescope

        evt: (run, evt, tel)
        """
        if evt in mapping:
            return mapping[evt]

        # Event not in mapping. Try again with neutral telescope
        evt = (*evt[:2], -1)
        return mapping.get(evt, None)

    @classmethod
    def subjects_from_files(cls, paths, mapping):
        subjects = {}
        for run, event, charge in cls.load_files(paths):
            evt = cls.parse_event(run, event)
            subject = cls.evt_to_subj(evt, mapping)

            if subject is not None:
                charge = charge[:-1]
                s = Subject(subject, evt, charge)

                subjects[subject] = s

        return subjects

    ##########################################################################
    ###   Plotting   #########################################################
    ##########################################################################

    def plot_sample(self, size, **kwargs):
        s = self._sample_s(size)
        return s.plot_subjects(plt.figure(), **kwargs)

    def plot_subjects(self, fig=None, w=5, camera=None):
        if camera is None:
            camera = Camera()

        if fig is None:
            fig = plt.figure()

        l = math.ceil(len(self.subjects) / w)
        fig.set_size_inches(2*w, 2*l)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05,
                            wspace=.05)

        axes = fig.subplots(l, w, True, subplot_kw={
            'xticks': [],
            'yticks': []
        })
        if l > 1:
            axes = axes.flatten()
        if w == 1:
            axes = [axes]
        for i, subject in enumerate(self.list()):
            subject.plot(axes[i], camera)

        fig.savefig('test.png')
        return fig

    ##########################################################################
    ###   Loading Data from HDF files   ######################################
    ##########################################################################

    @classmethod
    def parse_event(cls, run, evt):
        def parse(regex, string):
            s = regex.search(string)
            return int(s.group(1))

        run = parse(cls.patterns['run'], run)
        event = parse(cls.patterns['evt'], evt)
        tel = parse(cls.patterns['tel'], evt)

        return (run, event, tel)

    @classmethod
    def load_files(cls, args):
        print('loading files from %s' % str(args))
        paths = []
        for path in args:
            print(path)
            if os.path.isdir(path):
                for fname in os.listdir(path):
                    print(fname)
                    if os.path.splitext(fname)[1] == '.hdf5':
                        if path not in paths:
                            paths.append(os.path.join(path, fname))

            elif os.path.splitext(path)[1] == '.hdf5':
                if path not in paths:
                    paths.append(path)

        print('loading paths %s' % paths)
        for fname in paths:
            for item in cls.load_file(fname):
                yield item

    @staticmethod
    def load_file(fname):
        print('Loading subjects from %s' % fname)
        bar = progressbar.ProgressBar()
        with h5py.File(fname) as file:
            for run in file:
                # for event in bar(file[run]):
                for event in file[run]:
                    if event == 'summary':
                        continue
                    try:
                        charge = file[run][event]['charge']
                    except KeyError:
                        print(run, event)
                        raise
                    yield(run, event, charge)

    def scale_charges(self):
        order = []
        charges = np.zeros(self.dimensions)

        for i, subject in enumerate(self.list()):
            order.append(subject.id)
            charges[i] = np.array(subject.charge)

        charges = preprocessing.scale(charges)
        print(charges)
        for i, s in enumerate(order):
            self.subjects[s].scaled_charge = charges[i]

        return order, charges

    def get_charge_array(self, labels=False):
        _labels = labels
        subjects = self.list()
        order = []
        labels = []
        charges = np.zeros(self.dimensions)
        for i, subject in enumerate(subjects):
            order.append(subject.id)
            labels.append(subject.label)
            charges[i] = subject.scaled_charge

        if _labels:
            return order, charges, labels
        return order, charges

    ##########################################################################
    ###   Operator Overloading   #############################################
    ##########################################################################

    def __getitem__(self, subject):
        return self.subjects[subject]

    def __len__(self):
        return len(self.subjects)
