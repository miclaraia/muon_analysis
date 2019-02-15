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
                    CREATE INDEX image_subjects_id
                    ON image_subjects (image_id);


                CREATE TABLE IF NOT EXISTS image_groups (
                    group_id integer PRIMARY KEY,
                    image_size integer,
                    image_width integer,
                    description text,
                    permutations integer
                );




                CREATE TABLE IF NOT EXISTS subjects (
                    subject_id integer PRIMARY KEY, -- assigned subject id

                    /* source run, event, and telescope id
                       store as run_event_tel */
                    source_id text NOT NULL,
                    source TEXT NOT NULL, -- source filename

                    charge BLOB,
                    batch_id INTEGER, -- group subjects in batches
                    split_id INTEGER DEFAULT 0 -- which split group for training
                );
                    CREATE INDEX IF NOT EXISTS subject_batch
                        ON subjects (batch_id);
                    CREATE INDEX IF NOT EXISTS subject_split
                        ON subjects (split_id, batch_id);



                /*CREATE TABLE IF NOT EXISTS clustering (
                    subject_id integer PRIMARY KEY,
                    cluster integer,
                    is_test boolean DEFAULT 0,
                    split integer
                );*/

                CREATE TABLE IF NOT EXISTS subject_clusters (
                    subject_id INTEGER,
                    cluster_name TEXT NOT NULL,
                    cluster INTEGER
                );

                    CREATE INDEX IF NOT EXISTS id_cluster
                        ON subject_clusters (cluster_name, subject_id);

                CREATE TABLE IF NOT EXISTS subject_labels (
                    subject_id integer,
                    label_name text,
                    label integer
                );

                    CREATE INDEX IF NOT EXISTS subject_label_names 
                        ON subject_labels (label_name, subject_id);
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
        _splits = {k: i for i, k in \
                enumerate(['train', 'test', 'valid', 'train_dev'])}

        @classmethod
        def next_batch(cls, conn):
            cursor = conn.execute('SELECT MAX(batch_id) FROM subjects')
            next_id = cursor.fetchone()[0]
            if next_id:
                return next_id + 1
            return 0

        @classmethod
        def next_id(cls, conn):
            cursor = conn.execute('SELECT MAX(subject_id) FROM subjects')
            next_id = cursor.fetchone()[0]
            if next_id:
                return next_id + 1
            return 0

        @classmethod
        def add_subject(cls, conn, subject, batch_id, split_name):

            split_id = cls._splits[split_name]
            # subject_id, run, evt, tel, charge, cluster, label
            data = {
                'subject_id': subject.id,
                'source_id': subject.source_id,
                'source': subject.source,
                'batch_id': batch_id,
                'split_id': split_id,
                'charge': subject.x.tostring(),
            }
            keys, values = zip(*data.items())

            query = 'INSERT INTO subjects ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['?' for _ in range(len(keys))]))
            conn.execute(query, values)

            return subject.id

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


        ### Getting Subjects #########################

        @classmethod
        def get_subject(cls, conn, subject_id):
            cursor = conn.execute("""
                SELECT subject_id, charge FROM subjects
                WHERE subject_id=?""", (subject_id,))

            row = cursor.fetchone()
            return Subject(
                id=row[0],
                charge=np.fromstring(row[4], dtype=np.float32)
            )

        @classmethod
        def get_subject_batch(cls, conn, batch_id):
            cursor = conn.execute("""
                SELECT subject_id, charge FROM subjects
                WHERE batch_id=?""", (batch_id,))
            for row in cursor:
                yield Subject(
                    id=row[0],
                    charge=np.fromstring(row[4], dtype=np.float32)
                )

        @classmethod
        def get_subjects(cls, conn, subject_ids):
            for subject_id in tqdm(subject_ids):
                yield cls.get_subject(conn, subject_id)

        @classmethod
        def get_all_subjects(cls, conn):
            cursor = conn.execute(
                'SELECT subject_id, charge FROM subjects')
            for row in cursor:
                yield Subject(
                    id=row[0],
                    charge=np.fromstring(row[4], dtype=np.float32)
                )


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
            cursor = conn.execute('SELECT subject_id FROM subjects')
            return [row[0] for row in cursor]


        @classmethod
        def set_subject_split(cls, conn, subject_id, split_name):
            split_id = cls._splits[split_name]
            cursor = conn.execute("""
                UPDATE subjects SET split_id=?
                WHERE subject_id=?""", (split_id, subject_id)
            )

        @classmethod
        def get_split_subjects(cls, conn, split_name):
            split_id = cls._splits[split_name]
            cursor = conn.execute(
                'SELECT subject_id FROM subjects WHERE split_id=?',
                (split_id,))
            for row in cursor:
                yield row[0]

    class Clustering:

        @classmethod
        def add_subject_cluster(cls, conn, subject_id, cluster_name, cluster):
            conn.execute("""
                INSERT INTO subject_clusters
                (subject_id, cluster_name, cluster)
                VALUES (?,?,?)""", (subject_id, cluster_name, cluster))

        @classmethod
        def get_cluster_assignments(cls, conn, cluster_name):
            cursor = conn.execute("""
                SELECT subject_id, cluster FROM subject_clusters
                WHERE cluster_name=?""", (cluster_name,))

            clusters = {}
            for subject_id, cluster in tqdm(cursor):
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(subject_id)
            return clusters

        @classmethod
        def get_cluster_labels(cls, conn, cluster_name, label_name):
            cursor = conn.execute("""
                SELECT subject_clusters.cluster, 
                    subject_clusters.subject_id, subject_labels.label
                    FROM subject_clusters
                INNER JOIN subject_labels
                    ON subject_clusters.subject_id=subject_labels.subject_id
                WHERE cluster_subjects.cluster_name=? 
                    AND subject_labels.label_name=?
                ORDER BY subject_clusters.cluster ASC""",
                (cluster_name, label_name,))

            for row in cursor:
                yield row

        @classmethod
        def get_cluster_subjects(cls, conn, cluster_name, cluster):
            cursor = conn.execute("""
                SELECT subject_id FROM cluster_subjects
                WHERE cluster_name=? AND cluster=?""",
                (cluster_name, cluster))

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
            conn.execute('DELETE FROM image_groups WHERE group_id=?', (group_id,))
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

            query = 'INSERT INTO image_groups ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['?' for _ in range(len(keys))]))
            conn.execute(query, values)

            return group.group_id

        @classmethod
        def list_groups(cls, conn):
            cursor = conn.execute('SELECT group_id FROM image_groups')
            return [row[0] for row in cursor]

        @classmethod
        def get_group(cls, conn, group_id):
            cursor = conn.execute("""
                SELECT group_id, image_size, image_width,
                description, permutations FROM image_groups WHERE group_id=?""",
                (group_id,))
            group = cursor.fetchone()

            return ImageGroup(
                group_id=group[0],
                image_size=group[1],
                image_width=group[2],
                description=group[3],
                permutations=group[4],
                images=None)




