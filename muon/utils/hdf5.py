
import muon.utils.camera as cam
from swap.db import DB

import re
import numpy as np
import h5py
import os
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


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

    def get_sample(self, size):
        subjects = list(self.subjects.values())
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
                score = self.swap_scores[subject]
                if score.label in [0, 1]:
                    s = Subject(subject, evt, charge, score)
                    print(s)

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
            {'retired_as': {'$in': [0, 1]}},
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

    @classmethod
    def run(cls, subjects):
        subjects = subjects.get_sample(10000)
        order, charges = cls.build_charge_array(subjects)

        pca = PCA(n_components=2)
        X = pca.fit_transform(charges)
        print(X)

        x, y = zip(*X)
        c = [s.label for s in subjects]
        plt.scatter(x, y, c=c, s=2.5)
        plt.show()

    @staticmethod
    def get_charge(subject):
        return np.array(subject.charge)

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

        charges = preprocessing.scale(charges)

        return subject_order, charges






    

"""
1. open a hdf5 file
2. iterate through events in file
    filter out duplicate events/telescopes
3. find appropriate subject
4. build data array
5. run pca
6. plot results"""
