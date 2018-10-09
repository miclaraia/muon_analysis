import tqdm
import h5py
import json
import os
import csv
from tqdm import tqdm

from muon.subjects import Subject


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

    def add_subjects(self, subjects):
        skipped = []
        for subject in tqdm(subjects):
            skipped += self.add_subject(subject)
        return skipped

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
            return file

    def iter(self):
        hdf = self._file
        for subject in hdf['subjects']:
            yield self._to_subject(subject, hdf['subjects'[subject]])

    def _to_subject(self, id, hdf_subject):
        charge = hdf_subject['charge'][:]
        metadata = json.loads(hdf_subject.attrs['metadata'])
        label = json.loads(hdf_subject.attrs['label'])

        return Subject(id, charge, metadata, label)
