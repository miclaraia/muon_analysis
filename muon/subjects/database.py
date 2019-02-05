import sqlite3
import os
import numpy as np
import json

from muon.subjects.subject import Subject


class Database:

    def __init__(self, fname):
        self.fname = fname

        if not os.path.isfile(fname):
            self.create_db()

    @property
    def conn(self):
        return self.__class__.Connection(self.fname)
        # return sqlite3.connect(self.fname)


    def create_db(self):
        with self.conn as conn:

#         Tables:
#             Images
#                 image_id, group_id, metadata, zoo_id
#             ImageSubjects
#                 subject_id, image_id, image_location details
#             ImageGroups
#                 group_id, metadata->(size, dim, group, description)
            query = ''
            query = """
                CREATE TABLE IF NOT EXISTS images (
                    image_id integer PIMARY KEY,
                    group_id integer,
                    cluster integer,
                    metadata text NOT NULL,
                    zoo_id integer
                );

                CREATE TABLE IF NOT EXISTS image_subjects (
                    subject_id integer,
                    image_id integer,
                    group_id integer,
                    image_index integer
                );

                CREATE TABLE IF NOT EXISTS groups (
                    group_id integer PRIMARY KEY,
                    image_size integer,
                    image_width integer,
                    description text,
                    permutations integer
                );

                CREATE TABLE IF NOT EXISTS subjects (
                    subject_id integer PRIMARY KEY,
                    run integer,
                    evt integer,
                    tel integer,
                    charge BLOB,
                    cluster integer,
                    label integer
                );

                CREATE TABLE IF NOT EXISTS subject_labels (
                    subject_id integer,
                    label_name text,
                    label integer
                );
            """
            print(query)
            conn.executescript(query)

    class Connection:

        def __init__(self, fname):
            self.fname = fname
            self.conn = None

        def __enter__(self):
            self.conn = sqlite3.connect(self.fname)
            return self.conn

        def __exit__(self, type, value, traceback):
            self.conn.close()


class Subjects:

    @classmethod
    def next_id(cls, conn):
        cursor = conn.execute('SELECT MAX(subject_id) FROM subjects')
        return cursor.fetchone()[0] + 1

    @classmethod
    def add_subject(cls, conn, subject):
        # subject_id, run, evt, tel, charge, cluster, label
        subject.id = cls.next_id(conn)
        data = {
            'subject_id': subject.id,
            'run': subject.metadata['run'],
            'evt': subject.metadata['evt'],
            'tel': subject.metadata['tel'],
            'charge': np.getbuffer(subject.x),
        }
        keys, values = zip(*data.items())

        conn.execute('INSERT INTO subjects ? VALUES ?', (keys, values))
        return subject.id

    @classmethod
    def add_subjects(cls, conn, subjects):
        for subject in subjects:
            yield cls.add_subject(conn, subject)

    @classmethod
    def get_subject(cls, conn, subject_id):
        cursor = conn.execute(
            'SELECT subject_id, run, evt, tel, charge FROM subjects' \
            'WHERE subject_id=?', subject_id)

        row = cursor.fetchone()
        return Subject(
            id=row[0],
            metadata={
                'run': row[1],
                'evt': row[2],
                'tel': row[3]
            },
            charge=row[4]
        )


class Image:

    @classmethod
    def next_id(cls, conn):
        cursor = conn.execute('SELECT MAX(*) FROM images')
        return cursor.fetchone()[0] + 1

    @classmethod
    def add_image(cls, conn, image):
        image.image_id = cls.next_id(conn)
        data = {
            'image_id': image.image_id,
            'group_id': image.group_id,
            'cluster': image.cluster,
            'metadata': json.dumps(image.metadata),
            'zoo_id': image.zoo_id
        }
        keys, values = zip(*data.items())
        conn.execute('INSERT INTO images ? VALUEs ?', (keys, values))

