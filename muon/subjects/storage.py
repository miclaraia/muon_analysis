import numpy as np
from tqdm import tqdm
from uuid import uuid4
import sqlite3
import logging

from muon.subjects import Subject
from muon.subjects import Subjects

logger = logging.getLogger(__name__)


class Storage:

    def __init__(self, database):
        self.database = database

    @property
    def conn(self):
        return self.database.conn

    def add_batch(self, subjects, label_name=None):
        with self.conn as conn:
            next_batch = self.database.Subject.next_batch(conn)
        self.add_subjects(subjects, next_batch, label_name)

    def add_subjects(self, subjects, batch_id, label_name=None):

        with self.conn as conn:
            for subject in tqdm(subjects):
                self._add_subject(conn, subject, batch_id, label_name)

            # print(self.database.Subject.next_id(conn))
            # print(dir(self.database.Subject))
            # print(subjects, list(subjects))
            # self.database.Subject.add_subjects(conn, subjects)

            conn.commit()

    def add_subject(self, subject, batch_id, label_name=None):
        with self.conn as conn:
            self._add_subject(conn, subject, batch_id, label_name)
            conn.commit()

    def _add_subject(self, conn, subject, batch_id, label_name=None):

        existing_id = self.database.Subject \
            .get_source_subject(conn, subject.source_id)

        if not existing_id:
            logger.info('Subject ({}, {}) already in database, skipping'
                .format(existing_id, subject.source_id))
            return existing_id

        split_probs = {
            'test': 0.25,
            'train': 0.75
            # 'train': 0.75*0.8,
            # 'valid': 0.75*0.1,
            # 'train_dev': 0.75*0.1
        }

        k = list(split_probs.keys())
        p = list(split_probs.values())
        split = str(np.random.choice(k, 1, p=p)[0])

        subject.id = str(uuid4())
        try:
            self.database.Subject.add_subject(
                conn, subject, batch_id, split)
        except sqlite3.IntegrityError as e:
            logger.error(
                'Integrity error, retrying with new subject id')
            logger.exception(e)
            subject.id = str(uuid4())
            self._add_subject(conn, subject, batch_id)
            self.database.Subject.add_subject(
                conn, subject, batch_id, split)

        if label_name and subject.y is not None:
            self.database.Subject \
                .add_subject_label(subject.id, label_name, subject.y)

        return subject.id

    def add_labels(self, label_name, labels):
        skipped = []
        database = self.database

        with database.conn as conn:
            for subject_id, label in tqdm(labels):
                database.Subject.add_subject_label(
                    conn, subject_id, label_name, label)

            conn.commit()

    def list_label_names(self):
        with self.conn as conn:
            return self.database.Subject.list_label_names(conn)

    def get_subject(self, subject_id):
        with self.conn as conn:
            return self.database.Subject.get_subject(conn, subject_id)

    def get_all_subjects(self):
        with self.conn as conn:
            subjects = [s for s in tqdm(
                self.database.Subject.get_all_subjects(conn))]
            return Subjects(subjects)

    def get_subject_batches(self, batches):
        with self.conn as conn:
            subject_iter = self.database.Subject \
                .get_subject_batches(conn, batches)
            subjects = Subjects(list(subject_iter))
            return subjects

    def get_subjects(self, subject_ids):
        with self.conn as conn:
            subjects = []
            for subject_id in tqdm(subject_ids):
                subjects.append(
                    self.database.Subject.get_subject(conn, subject_id))

            return Subjects(subjects)

        subjects = [self.get_subject(s) for s in tqdm(subjects)]
        return Subjects(subjects)

    def get_split_subjects(self, split_name, batches=None):
        with self.conn as conn:
            if batches:
                subject_ids = self.database.Subject \
                    .get_split_subjects_batches(conn, split_name, batches)
            else:
                subject_ids = self.database.Subject \
                    .get_split_subjects(conn, split_name)

            return self.get_subjects(subject_ids)

    def get_subject_labels(self, subject_ids):
        with self.conn as conn:
            for subject_id in subject_ids:
                label = self.database.Subject \
                    .get_subject_label(conn, subject_id)
                yield subject_id, label

    def get_cluster_subjects(self, cluster):
        with self.conn as conn:
            subjects = self.database.Clustering. \
                get_cluster_subjects(conn, cluster)
            return self.get_subjects(subjects)

    def labeled_subjects(self, label_name):
        with self.conn as conn:
            return self.database.Subject.list_labeled_subjects(
                conn, label_name)


    # def reserve_test_set(cls, conn):
        # with self.conn as conn:
            # test_set = self.database.Subject. \
                # get_split_subjects(conn, 'test')
            # train_set = self.database.Subject. \
                # get_split_subjects(conn, 'train')
            # all_set = self.database.Subject.list_subjects(conn)

            # n_test = len(test_set)
            # n_avail = len(set(all_set) - set(test_set))

            # n_add = int((n_test + n_avail)*0.25 - n_test)
            # assert n_add >= 0
            # assert n_add >= len(train_set)

            # for subject_id in np.random.choice(train_set, n_add):
                # self.database.Subject. \
                    # set_subject_split(conn, subject_id, 'train'

                        # )
                # database.Clustering.set_subject_test_flag(
                    # conn, subject_id, True)

            # conn.commit()




    def iter(self):
        with self.conn as conn:
            for subject in tqdm(self.database.Subject.get_all_subjects(conn)):
                yield subject

    def split_assignment_counts(self):
        splits = {}
        with self.conn as conn:
            for split in self.database.Subject._splits:
                subjects = self.database.Subject \
                    .get_split_subjects(conn, split)
                splits[split] = len(list(subjects))

