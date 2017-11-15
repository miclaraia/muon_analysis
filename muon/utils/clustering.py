
import muon.utils.camera as cam
import muon.utils.subjects
from muon.utils.camera import Camera
from swap.db import DB

import re
import numpy as np
import h5py
import os
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pickle


class Subject:

    def __init__(self, subject, event, charge, score):
        self.id = subject
        self.event = event
        self.charge = np.array(charge)

        self.label = score.label
        self.score = score.p

    def __str__(self):
        return 'id %d event %s label %d score %f' % \
               (self.id, self.event, self.label, self.score)

class Subjects:

    patterns = {
        'run': re.compile('run([0-9]+)'),
        'evt': re.compile('evt([0-9]+)'),
        'tel': re.compile('tel([0-9]+)'),
    }

    def __init__(self, path=None):
        self.subject_mapping = self._make_subject_mapping()
        self.swap_scores = self.get_swap_scores()
        self.subjects = {}

        if path:
            self.subjects_from_files(path)

    def __getitem__(self, subject):
        return self.subjects[subject]

    def get_sample(self, size):
        size = int(size)
        subjects = list(self.subjects.values())
        print('number of subjects', len(subjects))
        if size > len(subjects):
            return subjects
        return random.sample(subjects, size)

    def evt_to_subj(self, evt):
        mapping = self.subject_mapping
        if evt in mapping:
            return mapping[evt]
        else:
            evt = list(evt)
            evt[2] = -1
            evt = tuple(evt)
            return mapping.get(evt, None)

    def subjects_from_files(self, paths):
        subjects = {}
        for run, event, charge in self.load_files(paths):
            evt = self.parse_event(run, event)
            subject = self.evt_to_subj(evt)

            if subject in self.swap_scores:
                charge = charge[:-1]
                score = self.swap_scores[subject]
                # if score.label in [0, 1]:
                s = Subject(subject, evt, charge, score)

                subjects[subject] = s

        self.subjects = subjects
        return subjects

    @classmethod
    def parse_event(cls, run, evt):
        def parse(regex, string):
            s = regex.search(string)
            return int(s.group(1))

        run = parse(cls.patterns['run'], run)
        event = parse(cls.patterns['evt'], evt)
        tel = parse(cls.patterns['tel'], evt)

        return (run, event, tel)

    @staticmethod
    def _make_subject_mapping():
        cursor = DB().subjects.collection.find(
            {'retired_as': {'$in': [-1, 0, 1]}},
            {'subject': 1, 'metadata': 1}
        )   

        data = {}
        for item in cursor:
            run = item['metadata']['run']
            evt = item['metadata']['evt']
            tel = item['metadata']['tel']
            subject = item['subject']

            data[(run, evt, tel)] = subject

        return data

    @staticmethod
    def get_swap_scores():
        return DB().subjects.get_scores()

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
                        paths.append(fname)

            elif os.path.splitext(path)[1] == '.hdf5':
                paths.append(path)

        print('loading paths %s' % paths)
        for fname in paths:
            for item in cls.load_file(fname):
                yield item

    @staticmethod
    def load_file(fname):
        with h5py.File(fname) as file:
            for run in file:
                for event in file[run]:
                    if event == 'summary':
                        continue
                    try:
                        charge = file[run][event]['charge']
                    except KeyError:
                        print(run, event)
                        raise
                    yield(run, event, charge)


class Cluster:

    figure = 0

    def __init__(self, pca, subjects, sample):
        self.pca = pca
        self.subjects = subjects
        self.sample_X = self.project_subjects(sample)

    @classmethod
    def create(cls, subjects):
        _subjects = list(subjects.subjects.values())
        _, charges = cls.build_charge_array(_subjects)

        pca = PCA(n_components=2)
        pca.fit(charges)

        sample = subjects.get_sample(1e4)
        return cls(pca, subjects, sample)

    @classmethod
    def run(cls, subjects):
        cluster = cls.create(subjects)
        cluster.plot()

        import code
        code.interact(local=locals())

    def count_class(self, bound, axis, direction):
        s = list(self.subjects.subjects.values())
        order, X = self.project_subjects(s)

        count = 0
        for item in X:
            if direction == 1:
                if item[axis] > bound:
                    count += 1
            else:
                if item[axis] < bound:
                    count += 1


    def visualize(self):
        camera = Camera()
        fig = plt.figure(figsize=(9, 1))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05,
                            wspace=.05)

        count = self.pca.n_components_
        for n in range(count):
            component = self.pca.components_[n, :]
            ax = fig.add_subplot(1, count+1, n+1, xticks=[], yticks=[])
            data = []
            for i, c in enumerate(component):
                x, y = camera.coordinates[i+1]
                data.append((x, y, c))

            x, y, c = zip(*data)
            ax.scatter(x, y, c=c, s=10, cmap='viridis')

        ax = fig.add_subplot(1, count+1, count+1, xticks=[], yticks=[])
        subjects = list(self.subjects.subjects.values())
        c = [s.charge for s in subjects]
        c = np.array(c)
        c = np.mean(c, axis=0)
        data = []
        for i, c in enumerate(c):
            x, y = camera.coordinates[i+1]
            data.append((x, y, c))

        x, y, c = zip(*data)
        ax.scatter(x, y, c=c, s=10, cmap='viridis')

        plt.show()

    def visualize_mean(self):
        camera = Camera()
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05,
                            wspace=.05)

        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        data = []

        subjects = list(self.subjects.subjects.values())
        c = [s.charge for s in subjects]
        c = np.array(c)
        c = np.mean(c, axis=0)
        for i, c in enumerate(c):
            x, y = camera.coordinates[i+1]
            data.append((x, y, c))

        x, y, c = zip(*data)
        ax.scatter(x, y, c=c, s=10, cmap='viridis')

        plt.show()


    def plot(self, save=False):
        subject_order, X = self.sample_X
        data = {k:[] for k in [-1, 0, 1]}
        for i, s in enumerate(subject_order):
            x, y = X[i]
            label = self.subjects[s].label
            c = self.color(label)
            data[label].append((x, y, c))

        def plot(v):
            x, y, c = zip(*data[v])
            plt.scatter(x, y, c=c, s=2.5)

        for i in [-1, 0, 1]:
            plot(i)

        plt.title('PCA dimensionality reduction of Muon Data')
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')

        # c = [s.label for s in subjects]
        if save:
            plt.savefig('Figure_%d' % self.figure)
            self.figure += 1
        else:
            plt.show()

    def download_plotted_subjects(self, x, y, c, size, prefix='', dir_=None):
        subjects = self.subjects_in_range(x, y, c, self.sample_X[0])
        self.download_subjects(subjects, size, prefix, dir_)

    def subjects_in_range(self, x, y, c, subjects=None):
        if subjects is None:
            subjects = list(self.subjects.subjects.values())

        # Remap bounding box coordinates so name doesn't conflict
        x_ = x
        y_ = y

        if type(c) is int:
            c = [c]

        def in_range(x, y):
            """
            Check if coordinates are inside the bounding box
            """
            def check(x, bounds):
                if bounds is None:
                    return True

                min, max = bounds
                if x > min and x < max:
                    return True
                return False

            return check(x, x_) and check(y, y_)

        def check_type(subject):
            """
            Check if subject is in the required class
            """
            if c is None:
                return True
            subject = self.subjects[subject]
            return subject.label in c

        order, X = self.project_subjects(subjects)
        subjects = []
        for i, point in enumerate(X):
            x, y = point
            if in_range(x, y):
                subject = order[i]
                if check_type(subject):
                    subjects.append(subject)

        return subjects

    def download_subjects(self, subjects, size=None, prefix='', dir_=None):
        """
        Download subject images from panoptes

        subjects: list of subject ids
        size: select random sample from list of subjects
        """
        if size is not None:
            subjects = random.sample(subjects, size)
        muon.utils.subjects.download_images(subjects, prefix, dir_)

    @staticmethod
    def color(value):
        if value == -1:
            return (.8, .8, .8)
        elif value == 0:
            return (.1, .1, .8)
        return (.9, .1, .1)

    @staticmethod
    def get_charge(subject):
        return np.array(subject.charge)

    def project_subjects(self, subjects):
        """
        subjects: list of subjects to project
        """
        order, charges = self.build_charge_array(subjects)
        X = self.pca.transform(charges)
        return order, X

    @classmethod
    def build_charge_array(cls, subjects):
        """
        subjects: list of subjects
        """
        subject_order = []
        charges = np.zeros((len(subjects), len(subjects[0].charge)))
        for i, subject in enumerate(subjects):
            subject_order.append(subject.id)
            charges[i] = cls.get_charge(subject)

        print(charges)
        charges = preprocessing.scale(charges)
        print(charges)

        return subject_order, charges






    

"""
1. open a hdf5 file
2. iterate through events in file
    filter out duplicate events/telescopes
3. find appropriate subject
4. build data array
5. run pca
6. plot results"""
