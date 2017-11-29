
from swap.db import DB

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


def _get_subject(subject_id):
    return pclient.subject.Subject(subject_id)


def _get_image_url(subject_id):
    s = _get_subject(subject_id)
    url = list(s.raw['locations'][0].values())[0]
    return url


def download_image(subject_id, prefix=None, dir_=None):
    if prefix is None:
        prefix = ''

    url = _get_image_url(subject_id)
    ext = url.split('.')[-1]

    fname = 'muon-%s-%d.%s' % (prefix, subject_id, ext)
    if dir_ is not None:
        fname = os.path.join(dir_, fname)

    urllib.request.urlretrieve(url, fname)


def download_images(subjects, prefix=None, dir_=None):
    for i in subjects:
        download_image(i, prefix, dir_)

def _load_images(subjects):
    urls = [_get_image_url(s) for s in subjects]
    for u in urls:
        yield imread(u)


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


class Subject:

    def __init__(self, subject, event, charge, score):
        self.id = subject
        self.event = event
        self.charge = np.array(charge)
        self.scaled_charge = None

        self.label = score.label
        self.score = score.p

        self._normalize()

    def download_image(self, prefix, dir_):
        download_image(self.id, prefix, dir_)

    def color(self):
        if self.label == -1:
            return (.8, .8, .8)
        elif self.label == 0:
            return (.1, .1, .8)
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
        return 'id %d event %s label %d score %f' % \
               (self.id, self.event, self.label, self.score)

class Subjects:

    _mapping = _make_subject_mapping()
    patterns = {
        'run': re.compile('run([0-9]+)'),
        'evt': re.compile('evt([0-9]+)'),
        'tel': re.compile('tel([0-9]+)'),
    }

    def __init__(self, subjects):
        if type(subjects) is list:
            subjects = {s.id:s for s in subjects}
        self.subjects = subjects

        self.dimensions = self._dimensions(self.list())

    @classmethod
    def from_data(cls, path):
        subject_data = cls.subjects_from_files(path)

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

    def list(self):
        return list(self.subjects.values())

    def subject_ids(self):
        return list(self.subjects.keys())

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

        return Subjects(subjects)

    @staticmethod
    def _dimensions(subjects):
        l = len(subjects)
        if l > 0:
            w = len(subjects[0].charge)
            return l, w
        return l,

    @classmethod
    def evt_to_subj(cls, evt):
        """
        Get subject associated with specific run event and telescope

        evt: (run, evt, tel)
        """
        mapping = cls._mapping
        if evt in mapping:
            return mapping[evt]

        # Event not in mapping. Try again with neutral telescope
        evt = (*evt[:2], -1)
        return mapping.get(evt, None)

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

    def download_images(self, prefix=None, dir_=None):
        for s in self.subjects.values():
            s.download_image(prefix, dir_)

    def load_images(self):
        return _load_images(self.subject_ids())

    @staticmethod
    def get_swap_scores():
        return DB().subjects.get_scores()

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
