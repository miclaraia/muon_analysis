import tqdm
import h5py
import json
import os
import csv
from tqdm import tqdm

from muon.subjects import Subject
from muon.subjects import Subjects
# TODO use some kind of database instead of hdf


class Storage:

    def __init__(self, fname):
        self.fname = fname
        self._f = None
        self._manifest = None

    @property
    def _file(self):
        if self._f is None:
            self._f = self.load()
        return self._f

    @property
    def manifest(self):
        if self._manifest is None:
            fname = os.path.splitext(self.fname)[0] + '.json'
            if not os.path.isfile(fname):
                self._manifest = {'subjects': {}, 'next_id': 0}
            else:
                with open(fname, 'r') as file:
                    self._manifest = json.load(file)

        return self._manifest

    def close(self):
        if self._f:
            self._f.close()

    def add_subjects(self, subjects):
        skipped = []
        for subject in tqdm(subjects):
            skipped += self.add_subject(subject)
        return skipped

    def add_labels(self, name, labels):
        skipped = []
        hdf = self._file
        for subject, label in tqdm(labels):
            if subject in hdf['subjects']:
                hdf_s = hdf['subjects'][subject]
                s_labels = json.loads(hdf_s.attrs['label'])
                if s_labels is None:
                    s_labels = {}
                s_labels[name] = label
                hdf_s.attrs['label'] = json.dumps(s_labels)
            else:
                skipped.append(subject)

        all_labels = json.loads(hdf.attrs['labels'])
        all_labels[name] = [s for s, l in labels]
        hdf.attrs['labels'] = json.dumps(all_labels)

        return skipped

    def list_label_names(self):
        return list(json.loads(self._file.attrs['labels']))
    
    def add_subject(self, subject):
        hdf = self._file
        if subject.id is None:
            subject.id = hdf.attrs['next_id']
            print('new_id:', subject.id)
            hdf.attrs['next_id'] = hdf.attrs['next_id'] + 1
        if str(subject.id) in hdf['subjects']:
            print('Skipping subject, already in file: {}'.format(subject))
            return [subject.id]
        if subject.id > hdf.attrs['next_id']:
            hdf.attrs['next_id'] = subject.id + 1

        # print(subject.id, hdf.attrs['next_id'])

        self._add_subject(subject)
        return []

    def _add_subject(self, subject):
        hdf = self._file
        
        group = hdf.create_group('subjects/{}'.format(subject.id))

        charge = subject.x
        dset = group.create_dataset('charge', charge.shape, dtype='f')
        dset[:] = charge

        group.attrs['metadata'] = json.dumps(subject.metadata)
        group.attrs['label'] = json.dumps(subject.y)


    # def add_manifest(self, subject):
        # manifest = self.manifest
        # manifest['next_id'] = next_id + 1
        # self.manifest['subjects'][subject] = {
            # 'run': subject.metadata['run'],
            # 'evt': subject.metadata['evt'],
            # 'tel': subject.metadata['tel']}

    # @staticmethod
    # def _add_run(hdf_root, run):
        # return hdf_root.create_group('runs/{}'.format(run))

    # @staticmethod
    # def _add_evt(hdf_run, evt):
        # return hdf_run.create_group('evts/{}'.format(evt))

    # def ensure_subject_path(self, hdf, subject):
        # run = subject.metadata['run']
        # evt = subject.metadata['evt']

        # if run not in hdf['runs']:
            # self._add_run(hdf, run)
        # if evt not in hdf['runs'][run]['evts']:
            # self._add_evt(hdf['runs'][run], evt)

    @classmethod
    def new(cls, fname):
        pass

    def load(self):
        if os.path.isfile(self.fname):
            return h5py.File(self.fname, 'r+')
        else:
            file = h5py.File(self.fname, 'w')
            file.create_group('subjects')
            file.attrs['next_id'] = 0
            file.attrs['labels'] = json.dumps({})
            return file

    def get_subject(self, id):
        subject = self._file['subjects'][id]
        return self._to_subject(id, subject)

    def get_all_subjects(self):
        return Subjects([self.get_subject(s) for s in self._file['subjects']])

    def get_subjects(self, subjects):
        subjects = [self.get_subject(s) for s in tqdm(subjects)]
        return Subjects(subjects)

    def labeled_subjects(self, name):
        return json.loads(self._file.attrs['labels'])[name]

    def iter(self):
        hdf = self._file
        for subject in tqdm(hdf['subjects']):
            yield self._to_subject(subject, hdf['subjects'][subject])

    def to_subjects(self):
        return Subjects(list(self.iter()))

    def _to_subject(self, id, hdf_subject):
        charge = hdf_subject['charge'][:-1]
        metadata = json.loads(hdf_subject.attrs['metadata'])
        label = json.loads(hdf_subject.attrs['label'])

        return Subject(id, charge, metadata, label)
