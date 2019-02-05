import tqdm
from tqdm import tqdm

from muon.subjects import Subject
from muon.subjects import Subjects
# TODO use some kind of database instead of hdf


class Storage:

    def __init__(self, database):
        self.database = database

    def add_subjects(self, subjects):
        with self.database.conn as conn:
            next_id = self.database.Subject.next_id(conn)

            for subject in tqdm(subjects):
                subject.id = next_id
                self.database.Subject.add_subject(conn, subject)

                next_id += 1


            # print(self.database.Subject.next_id(conn))
            # print(dir(self.database.Subject))
            # print(subjects, list(subjects))
            # self.database.Subject.add_subjects(conn, subjects)
            conn.commit()

    def add_subject(self, subject):
        with self.database.conn as conn:
            subject.id = self.database.Subject.next_id(conn)
            self.database.Subject.add_subject(conn, subject)
            conn.commit()

    def add_labels(self, label_name, labels):
        skipped = []
        database = self.database

        with database.conn as conn:
            for subject, label in tqdm(labels):
                database.Subject.add_subject_label(
                    conn, subject, label_name, label)

            conn.commit()

    def list_label_names(self):
        with self.database.conn as conn:
            return self.database.Subject.list_label_names(conn)

    def get_subject(self, subject_id):
        with self.database.conn as conn:
            return self.database.Subject.get_subject(conn, subject_id)

    def get_all_subjects(self):
        with self.database.conn as conn:
            subjects = [s for s in tqdm(
                self.database.Subject.get_all_subjects(conn))]
            return Subjects(subjects)

    def get_subjects(self, subject_ids):
        with self.database.conn as conn:
            subjects = []
            for subject_id in subject_ids:
                subjects.append(
                    self.database.Subject.get_subject(conn, subject_id))

            return Subjects(subjects)

        subjects = [self.get_subject(s) for s in tqdm(subjects)]
        return Subjects(subjects)

    def labeled_subjects(self, label_name):
        with self.database.conn as conn:
            return self.database.Subject.list_labeled_subjects(
                conn, label_name)

    def iter(self):
        with self.database.conn as conn:
            for subject in tqdm(self.database.Subject.get_all_subjects(conn)):
                yield subject

