import sqlite3
import os
import numpy as np
import json
from tqdm import tqdm

from muon.subjects.subject import Subject
from muon.subjects.images import Image, ImageGroup



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

                CREATE TABLE IF NOT EXISTS clustering (
                    subject_id integer PRIMARY KEY,
                    cluster integer,
                    is_test boolean DEFAULT 0,
                    split integer
                );

                CREATE TABLE IF NOT EXISTS subject_labels (
                    subject_id integer,
                    label_name text,
                    label integer
                );

                CREATE INDEX subject_label_names 
                ON subject_labels (subject_id, label_name);

                CREATE INDEX clusters
                ON clustering (cluster)

				CREATE INDEX image_subjects_id
				ON image_subjects (image_id);
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


    class Subject:

        @classmethod
        def next_id(cls, conn):
            cursor = conn.execute('SELECT MAX(subject_id) FROM subjects')
            next_id = cursor.fetchone()[0]
            if next_id:
                return next_id + 1
            return 0

        @classmethod
        def add_subject(cls, conn, subject):
            # subject_id, run, evt, tel, charge, cluster, label
            data = {
                'subject_id': subject.id,
                'run': subject.metadata['run'],
                'evt': subject.metadata['evt'],
                'tel': subject.metadata['tel'],
                'charge': subject.x.tostring(),
            }
            keys, values = zip(*data.items())

            query = 'INSERT INTO subjects ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['?' for _ in range(len(keys))]))
            conn.execute(query, values)

            conn.execute(
                'INSERT INTO clustering (subject_id) VALUES (?)',
                (subject.id,))
            return subject.id

        @classmethod
        def add_subjects(cls, conn, subjects):
            raise Exception
            print(100)
            print(subjects)
            for subject in tqdm(subjects):
                print(subject)
                yield cls.add_subject(conn, subject)

        @classmethod
        def add_subject_label(cls, conn, subject_id, label_name, label):
            data = {
                'subject_id': subject_id,
                'label_name': label_name,
                'label': label
            }
            keys, values = zip(*data.items())

            query = 'INSERT INTO subject_labels ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['?' for _ in range(len(keys))]))
            conn.execute(query, values)

        @classmethod
        def get_subject_label(cls, conn, subject_id, label_name):
            cursor = conn.execute("""
                SELECT label FROM subject_labels WHERE
                subject_id=? AND label_name=?""", (subject_id, label_name))
            return cursor.fetchone()[0]

        @classmethod
        def list_label_names(cls, conn):
            cursor = conn.execute(
                'SELECT label_name FROM subject_labels GROUP BY label_name')
            return [row[0] for row in cursor]

        @classmethod
        def list_labeled_subjects(self, conn, label_name):
            cursor = conn.execute(
                'SELECT subject_id FROM subject_labels WHERE label_name=?',
                label_name)
            return [row[0] for row in cursor]

        @classmethod
        def list_subjects(self, conn):
            cursor = conn.execute(
                'SELECT subject_id FROM subject_labels',
                label_name)
            return [row[0] for row in cursor]

        @classmethod
        def get_subject(cls, conn, subject_id):
            cursor = conn.execute("""
                SELECT subject_id, run, evt, tel, charge FROM subjects
                WHERE subject_id=?""", (subject_id,))

            row = cursor.fetchone()
            return Subject(
                id=row[0],
                metadata={
                    'run': row[1],
                    'evt': row[2],
                    'tel': row[3]
                },
                    charge=np.fromstring(row[4], dtype=np.float32)
            )

        @classmethod
        def get_all_subjects(cls, conn):
            cursor = conn.execute(
                'SELECT subject_id, run, evt, tel, charge FROM subjects')
            for row in cursor:
                yield Subject(
                    id=row[0],
                    metadata={
                        'run': row[1],
                        'evt': row[2],
                        'tel': row[3]
                    },
                    charge=np.fromstring(row[4], dtype=np.float32)
                )

    class Clustering:

        @classmethod
        def set_subject_test_flag(cls, conn, subject_id, is_test_set):
            cursor = conn.execute("""
                UPDATE clustering SET is_test=?
                WHERE subject_id=?""", (is_test_set, subject_id)
            )

        @classmethod
        def set_subject_cluster(cls, conn, subject_id, cluster):
            conn.execute("""
                UPDATE clustering SET cluster=?
                WHERE subject_id=?""", (int(cluster), subject_id))

        @classmethod
        def get_cluster_assignments(cls, conn):
            cursor = conn.execute('SELECT subject_id, cluster FROM clustering')
            clusters = {}
            for subject_id, cluster in tqdm(cursor):
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(subject_id)
            return clusters

        @classmethod
        def get_cluster_labels(cls, conn, label_name):
            cursor = conn.execute("""
                SELECT clustering.cluster, 
                    clustering.subject_id, subject_labels.label
                    FROM clustering
                INNER JOIN subject_labels
                    ON clustering.subject_id=subject_labels.subject_id
                WHERE subject_labels.label_name=?
                ORDER BY clustering.cluster ASC""",
                (label_name,))
            
            for row in cursor:
                yield row

        @classmethod
        def get_cluster_subjects(cls, conn, cluster):
            cursor = conn.execute(
                'SELECT subject_id FROM clustering WHERE cluster=?',
                (cluster,))
            for row in cursor:
                yield row[0]

        @classmethod
        def get_subjects(cls, conn, is_test=False):
            cursor = conn.execute(
                'SELECT subject_id FROM clustering WHERE is_test=?',
                (int(is_test),))

            for row in cursor:
                yield row[0]

    class Image:

        @classmethod
        def next_id(cls, conn):
            cursor = conn.execute('SELECT MAX(image_id) FROM images')
            next_id = cursor.fetchone()[0]
            if next_id:
                return next_id + 1
            return 0

        @classmethod
        def add_image(cls, conn, image):
            data = {
                'image_id': image.image_id,
                'group_id': image.group_id,
                'cluster': image.cluster,
                'metadata': json.dumps(image.metadata),
                'zoo_id': image.zoo_id
            }
            keys, values = zip(*data.items())

            query = 'INSERT INTO images ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['?' for _ in range(len(keys))]))
            conn.execute(query, values)

            for i, subject_id in enumerate(image.subjects):
                conn.execute("""
                    INSERT INTO image_subjects
                    (subject_id, image_id, group_id, image_index)
                    VALUES (?,?,?,?)""",
                    (subject_id, image.image_id, image.group_id, i))

            return image.image_id

        @classmethod
        def get_image(cls, conn, image_id):
            cursor = conn.execute("""
                SELECT image_id, group_id, cluster, metadata, zoo_id
                FROM images WHERE image_id=?""", (image_id,))
            image = cursor.fetchone()

            cursor = conn.execute("""
                SELECT subject_id FROM image_subjects 
                WHERE image_id=?""", (image_id,))
            subjects = [row[0] for row in cursor]

            return Image(
                image_id=image[0],
                group_id=image[1],
                cluster=image[2],
                metadata=json.loads(image[3]),
                zoo_id=image[4],
                subjects=subjects)

        @classmethod
        def get_group_images(cls, conn, group_id):
            cursor = conn.execute(
                'SELECT image_id FROM images WHERE group_id=?', (group_id,))
            for row in cursor:
                yield cls.get_image(conn, row[0])

    class ImageGroup:
        # TODO add update methods

        @classmethod
        def next_id(cls, conn):
            cursor = conn.execute('SELECT MAX(group_id) FROM groups')
            next_id = cursor.fetchone()[0]
            if next_id:
                return next_id + 1
            return 0

        @classmethod
        def delete_group(cls, conn, group_id):
            print('Deleting group:', group_id, type(group_id))
            conn.execute('DELETE FROM groups WHERE group_id=?', (group_id,))
            conn.execute('DELETE FROM images WHERE group_id=?', (group_id,))
            conn.execute('DELETE FROM image_subjects WHERE group_id=?', (group_id,))

        @classmethod
        def add_group(cls, conn, group):

            data = {
                'group_id': group.group_id,
                'image_size': group.image_size,
                'image_width': group.image_width,
                'description': group.description,
                'permutations': group.permutations
            }
            keys, values = zip(*data.items())

            query = 'INSERT INTO groups ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['?' for _ in range(len(keys))]))
            conn.execute(query, values)

            return group.group_id

        @classmethod
        def list_groups(cls, conn):
            cursor = conn.execute('SELECT group_id FROM groups')
            return [row[0] for row in cursor]

        @classmethod
        def get_group(cls, conn, group_id):
            cursor = conn.execute("""
                SELECT group_id, image_size, image_width,
                description, permutations FROM groups WHERE group_id=?""",
                (group_id,))
            group = cursor.fetchone()

            return ImageGroup(
                group_id=group[0],
                image_size=group[1],
                image_width=group[2],
                description=group[3],
                permutations=group[4],
                images=None)




