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
                    zoo_id integer,
                    fig_dpi integer,
                    fig_offset integer,
                    fig_height integer,
                    fig_width integer,
                    fig_rows integer,
                    fig_cols integer
                );

                CREATE TABLE IF NOT EXISTS image_subjects (
                    subject_id TEXT NOT NULL,
                    image_id integer,
                    group_id integer,
                    image_index integer
                );
                    CREATE INDEX image_subjects_id
                    ON image_subjects (image_id, image_index);


                CREATE TABLE IF NOT EXISTS image_groups (
                    group_id integer PRIMARY KEY,
                    cluster_name TEXT NOT NULL,
                    image_size integer,
                    image_width integer,
                    description text,
                    permutations integer
                );




                CREATE TABLE IF NOT EXISTS subjects (
                    subject_id TEXT PRIMARY KEY, -- assigned subject id

                    /* source run, event, and telescope id
                       store as run_event_tel */
                    source_id text NOT NULL,
                    source TEXT NOT NULL, -- source filename

                    charge BLOB,
                    batch_id INTEGER, -- group subjects in batches
                    split_id INTEGER DEFAULT 0 -- which split group for training
                );
                    CREATE INDEX IF NOT EXISTS subject_batch
                        ON subjects (batch_id, split_id);
                    CREATE INDEX IF NOT EXISTS subject_split
                        ON subjects (split_id);
                    CREATE INDEX IF NOT EXISTS subject_source_id
                        ON subjects (source_id);
                    CREATE INDEX IF NOT EXISTS subject_source
                        ON subjects (source);



                /*CREATE TABLE IF NOT EXISTS clustering (
                    subject_id TEXT PRIMARY KEY,
                    cluster integer,
                    is_test boolean DEFAULT 0,
                    split integer
                );*/

                CREATE TABLE IF NOT EXISTS subject_clusters (
                    subject_id TEXT NOT NULL,
                    cluster_name TEXT NOT NULL,
                    cluster INTEGER
                );

                    CREATE INDEX IF NOT EXISTS id_cluster
                        ON subject_clusters (cluster_name, subject_id);

                CREATE TABLE IF NOT EXISTS subject_labels (
                    subject_id TEXT NOT NULL,
                    label_name text,
                    label integer
                );

                    CREATE INDEX IF NOT EXISTS subject_label
                        ON subject_labels (subject_id);
                    CREATE INDEX IF NOT EXISTS subject_label_names
                        ON subject_labels (label_name, subject_id);

                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    hash TEXT NOT NULL,
                    location TEXT NOT NULL,
                    updated TEXT NOT NULL
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


    class Subject:
        _splits = {k: i for i, k in \
                enumerate(['train', 'test', 'valid', 'train_dev'])}

        @classmethod
        def next_batch(cls, conn):
            cursor = conn.execute('SELECT MAX(batch_id) FROM subjects')
            last_id = cursor.fetchone()[0]
            if last_id is None:
                return 0
            return last_id + 1

        # @classmethod
        # def next_id(cls, conn):
            # cursor = conn.execute('SELECT MAX(subject_id) FROM subjects')
            # last_id = cursor.fetchone()[0]
            # if last_id is None:
                # return 0
            # return last_id + 1

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
                charge=np.fromstring(row[1], dtype=np.float32)
            )

        @classmethod
        def get_subject_batch(cls, conn, batch_id):
            cursor = conn.execute("""
                SELECT subject_id, charge FROM subjects
                WHERE batch_id=?""", (batch_id,))
            for row in cursor:
                yield Subject(
                    id=row[0],
                    charge=np.fromstring(row[1], dtype=np.float32)
                )

        @classmethod
        def get_subject_batches(cls, conn, batches):
            for batch_id in batches:
                for row in cls.get_subject_batch(conn, batch_id):
                    yield row

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
                    charge=np.fromstring(row[1], dtype=np.float32)
                )


        @classmethod
        def get_subject_label(cls, conn, subject_id, label_name):
            cursor = conn.execute("""
                SELECT label FROM subject_labels WHERE
                subject_id=? AND label_name=?""", (subject_id, label_name))
            return cursor.fetchone()[0]

        @classmethod
        def get_source_subject(cls, conn, source_id):
            cursor = conn.execute("""
                SELECT subject_id FROM subjects
                WHERE source_id=?""", (source_id,))
            row = cursor.fetchone()
            if row:
                return row[0]

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

        @classmethod
        def get_split_subjects_batch(cls, conn, split_name, batch_id):
            split_id = cls._splits[split_name]
            cursor = conn.execute("""
                SELECT subject_id FROM subjects
                WHERE split_id=? and batch_id=?
                """, (split_id, batch_id))
            for row in cursor:
                yield row[0]

        @classmethod
        def get_split_subjects_batches(cls, conn, split_name, batches):
            for batch in batches:
                for row in cls.get_split_subjects_batch(
                        conn, split_name, batch):
                    yield row

    class Clustering:

        @classmethod
        def add_subject_cluster(cls, conn, subject_id, cluster_name, cluster):
            conn.execute("""
                DELETE FROM subject_clusters
                WHERE subject_id=? AND cluster_name=?""",
                (subject_id, cluster_name))

            conn.execute("""
                INSERT INTO subject_clusters
                (subject_id, cluster_name, cluster)
                VALUES (?,?,?)""", (subject_id, cluster_name, cluster))

        @classmethod
        def get_cluster_assignments(cls, conn, cluster_name, batch=None):
            if batch:
                cursor = conn.execute("""
                    SELECT subject_clusters.subject_id, subject_clusters.cluster
                    FROM subject_clusters
                    INNER JOIN subjects
                        ON subject_clusters.subject_id=subjects.subject_id
                    WHERE
                        cluster_name=?
                        AND subjects.batch_id=?
                    """, (cluster_name, batch))
            else:
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
                WHERE subject_clusters.cluster_name=? 
                    AND subject_labels.label_name=?
                ORDER BY subject_clusters.cluster ASC""",
                (cluster_name, label_name,))

            for row in cursor:
                yield row

        @classmethod
        def get_cluster_subjects(cls, conn, cluster_name, cluster):
            cursor = conn.execute("""
                SELECT subject_id FROM subject_clusters
                WHERE cluster_name=? AND cluster=?""",
                (cluster_name, cluster))

            for row in cursor:
                yield row[0]

    class Image:

        @classmethod
        def next_id(cls, conn):
            cursor = conn.execute('SELECT MAX(image_id) FROM images')
            last_id = cursor.fetchone()[0]
            if last_id is None:
                return 2000
            return last_id + 1

        @classmethod
        def add_image(cls, conn, image):
            data = {
                'image_id': image.image_id,
                'group_id': image.group_id,
                'cluster': image.cluster,
                'metadata': json.dumps(image.metadata),
                'zoo_id': image.zoo_id
            }

            if image.image_meta is not None:
                data.update({
                    'fig_dpi': image.image_meta.dpi,
                    'fig_offset': image.image_meta.offset,
                    'fig_height': image.image_meta.height,
                    'fig_width': image.image_meta.width,
                    'fig_rows': image.image_meta.rows,
                    'fig_cols': image.image_meta.cols,
                })
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
        def update_zooid(cls, conn, image):
            conn.execute("""
                UPDATE images 
                SET zoo_id=?
                WHERE image_id=?""", (image.zoo_id, image.image_id))

        @classmethod
        def update_figure(cls, conn, image):
            args = (json.dumps(image.metadata),
                    image.image_meta.dpi,
                    image.image_meta.offset,
                    image.image_meta.height,
                    image.image_meta.width,
                    image.image_meta.rows,
                    image.image_meta.cols,
                    image.image_id)
            conn.execute("""
                UPDATE images
                SET
                    metadata=?,
                    fig_dpi=?,
                    fig_offset=?,
                    fig_height=?,
                    fig_width=?,
                    fig_rows=?,
                    fig_cols=?
                WHERE image_id=?""", args)
            

        @classmethod
        def get_image(cls, conn, image_id):
            cursor = conn.execute("""
                SELECT image_id, group_id, cluster, metadata, zoo_id,
                       fig_dpi, fig_offset, fig_height, fig_width,
                       fig_rows, fig_cols
                FROM images WHERE image_id=?""", (image_id,))
            image = cursor.fetchone()

            cursor = conn.execute("""
                SELECT subject_id FROM image_subjects 
                WHERE image_id=?
                ORDER BY image_index ASC""", (image_id,))
            subjects = [row[0] for row in cursor]

            image_meta = Image.ImageMeta(
                dpi=image[5],
                offset=image[6],
                height=image[7],
                width=image[8],
                rows=image[9],
                cols=image[10])

            return Image(
                image_id=image[0],
                group_id=image[1],
                cluster=image[2],
                metadata=json.loads(image[3]),
                zoo_id=image[4],
                subjects=subjects,
                image_meta=image_meta)

        @classmethod
        def get_group_images(cls, conn, group_id):
            cursor = conn.execute(
                'SELECT image_id FROM images WHERE group_id=?', (group_id,))
            for row in cursor:
                yield row[0]

        @classmethod
        def get_image_subjects(cls, conn, image_id):
            cursor = conn.execute("""
                SELECT subject_id FROM image_subjects
                WHERE image_id=?
                ORDER BY image_index ASC""", (image_id,))
            for row in cursor:
                yield row[0]

    class ImageGroup:
        # TODO add update methods

        @classmethod
        def next_id(cls, conn):
            cursor = conn.execute('SELECT MAX(group_id) FROM image_groups')
            last_id = cursor.fetchone()[0]
            if last_id is None:
                return 10
            return last_id + 1

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
                'cluster_name': group.cluster_name,
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
                SELECT group_id, cluster_name, image_size, image_width,
                description, permutations FROM image_groups WHERE group_id=?""",
                (group_id,))
            group = cursor.fetchone()

            return ImageGroup(
                group_id=group[0],
                cluster_name=group[1],
                image_size=group[2],
                image_width=group[3],
                description=group[4],
                permutations=group[5],
                images=None)


    class Source:

        @classmethod
        def add_source(cls, conn, source):
            data = {
                'source_id': source.source_id,
                'hash': source.hash,
                'location': source.fname,
                'updated': source.updated,
            }

            keys, values = zip(*data.items())

            query = 'INSERT INTO sources ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['?' for _ in range(len(keys))]))
            conn.execute(query, values)

        @classmethod
        def get_source(cls, conn, source_id):
            cursor = conn.execute("""
                SELECT source_id, hash, location, updated
                FROM sources
                WHERE source_id=?""", (source_id))

            row = cursor.fetchone()
            fields = ['source_id', 'hash', 'location', 'updated']
            return Source(**{k: row[i] for i, k in enumerate(fields)})







