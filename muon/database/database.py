import sqlite3
import os
import numpy as np
import json
from tqdm import tqdm

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
                    image_count integer,
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
                        ON subjects (batch_id, subject_id);
                    CREATE INDEX IF NOT EXISTS subject_batch_split
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
                    updated TIMESTAMP NOT NULL
                );
            """
            print(query)
            conn.executescript(query)

    class Connection:

        def __init__(self, fname):
            self.fname = fname
            self.conn = None

        def __enter__(self):
            self.conn = sqlite3.connect(self.fname, timeout=60)
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
            conn.execute("""
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
        def update_image(cls, conn, image_id, updates):
            query_args = []
            args = []
            for k in updates:
                query_args.append('{}=?'.format(k))
                args.append(updates[k])
            query = """
                UPDATE images
                SET {}
                WHERE image_id=?
                """.format(','.join(query_args))

            args.append(image_id)
            conn.execute(query, args)

        # @classmethod
        # def get_image(cls, conn, image_id):
            # fields = [
                # 'group_id',
                # 'cluster',
                # 'metadata',
                # 'zoo_id',
                # 'fig_dpi',
                # 'fig_offset',
                # 'fig_height',
                # 'fig_width',
                # 'fig_rows',
                # 'fig_cols'
                # ]

            # cursor = conn.execute("""
                # SELECT {}
                # FROM images WHERE image_id=?""".format(','.join(fields)),
                # (image_id,))
            # row = cursor.fetchone()
            # row = {fields[i]: v for i, v in enumerate(row)}

            # image_attrs = {}
            # fig_attrs = {}
            # for field in fields:
                # if 'fig_' in field:
                    # fig_attrs[field[4:]] = row[field]
                # elif field == 'metadata':
                    # image_attrs[field] = json.loads(row[field])
                # else:
                    # image_attrs[field] = row[field]

            # image_attrs['image_meta'] = fig_attrs
            # return image_attrs

        @classmethod
        def _parse_image_row(cls, row):
            fields = [
                'image_id',
                'group_id',
                'cluster',
                'metadata',
                'zoo_id',
                'fig_dpi',
                'fig_offset',
                'fig_height',
                'fig_width',
                'fig_rows',
                'fig_cols',
                'subjects'
                ]
            kwargs = {f: row[i] for i, f in enumerate(fields)}
            kwargs['subjects'] = row[-1].split(',')

            image_attrs = {}
            fig_attrs = {}
            for field in kwargs:
                if 'fig_' in field:
                    fig_attrs[field[4:]] = kwargs[field]
                elif field == 'metadata':
                    image_attrs[field] = json.loads(kwargs[field])
                else:
                    image_attrs[field] = kwargs[field]

            image_attrs['image_meta'] = fig_attrs

            return image_attrs

        @classmethod
        def _build_get_query(cls, where=None, exclude_zoo=False, shuffle=False):
            if where is None:
                where = 'image_id'
            if where == 'image_id':
                where = 'WHERE images.image_id=?'
            elif where == 'group_id':
                where = 'WHERE images.group_id=?'
                if exclude_zoo:
                    where += ' AND image.zoo_id IS NULL'

            if shuffle:
                order = 'ORDER BY RANDOM()'
            else:
                order = ''

            fields = ','.join([
                'images.image_id',
                'images.group_id',
                'cluster',
                'metadata',
                'zoo_id',
                'fig_dpi',
                'fig_offset',
                'fig_height',
                'fig_width',
                'fig_rows',
                'fig_cols',
                'GROUP_CONCAT(image_subjects.subject_id)'
                ])

            query = """
                SELECT {fields}
                FROM images JOIN image_subjects ON
                    images.image_id=image_subjects.image_id
                {where}
                GROUP BY images.image_id
                {order}
            """.format(fields=fields, where=where, order=order)

            return query

        @classmethod
        def get_images(cls, conn, image_ids):
            query = cls._build_get_query(where='image_id')
            cursor = conn.executemany(query, (image_ids,))
            for row in cursor:
                yield cls._parse_image_row(row)

        @classmethod
        def get_image(cls, conn, image_id):
            query = cls._build_get_query(where='image_id')
            cursor = conn.execute(query, (image_id,))
            row = cursor.fetchone()
            return cls._parse_image_row(row)

        @classmethod
        def get_group_images(cls, conn, group_id,
                             exclude_zoo=False, shuffle=False):
            query = cls._build_get_query(
                where='group_id', exclude_zoo=exclude_zoo)
            cursor = conn.execute(query, (group_id,))
            for row in cursor:
                yield cls._parse_image_row(row)

        @classmethod
        def get_group_image_ids(cls, conn, group_id,
                                exclude_zoo=False, shuffle=False):
            query = """
                SELECT image_id FROM images
                WHERE group_id=?
            """
            if exclude_zoo:
                query += " AND zoo_id IS NULL"

            cursor = conn.execute(query, (group_id,))
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
                'image_count': group.image_count,
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
        def update_group(cls, conn, group_id, updates):
            query_args = []
            args = []
            for k in updates:
                query_args.append('{}=?'.format(k))
                args.append(updates[k])
            query = """
                UPDATE image_groups
                SET {}
                WHERE group_id=?
                """.format(','.join(query_args))

            args.append(group_id)
            conn.execute(query, args)

        @classmethod
        def list_groups(cls, conn):
            cursor = conn.execute('SELECT group_id FROM image_groups')
            return [row[0] for row in cursor]

        @classmethod
        def get_group(cls, conn, group_id):
            fields = [
                'image_count',
                'cluster_name',
                'image_size',
                'image_width',
                'description',
                'permutations']

            cursor = conn.execute("""
                SELECT {}
                FROM image_groups WHERE group_id=?""".format(','.join(fields)),
                (group_id,))
            row = cursor.fetchone()
            row = {fields[i]: v for i, v in enumerate(row)}
            return row

            # return ImageGroup(
                # group_id=group[0],
                # cluster_name=group[1],
                # image_size=group[2],
                # image_width=group[3],
                # description=group[4],
                # permutations=group[5],
                # images=None)


    class Source:

        @classmethod
        def add_source(cls, conn, source):
            data = {
                'source_id': source.source_id,
                'hash': source.hash,
                'updated': source.updated,
            }

            keys, values = zip(*data.items())

            query = 'INSERT INTO sources ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['?' for _ in range(len(keys))]))
            conn.execute(query, values)

        @classmethod
        def get_source(cls, conn, source_id):
            fields = ['hash', 'updated']

            cursor = conn.execute("""
                SELECT {}
                FROM sources WHERE source_id=?""".format(','.join(fields)),
                (source_id,))
            row = cursor.fetchone()
            row = {fields[i]: v for i, v in enumerate(row)}
            return row

        @classmethod
        def source_exists(cls, conn, source_id):
            cursor = conn.execute("""
                SELECT source_id FROM sources
                WHERE source_id=?""", (source_id,))
            return cursor.fetchone() is not None

        @classmethod
        def update_source(cls, conn, source_id, updates):
            query_args = []
            args = []
            for k in updates:
                query_args.append('{}=?'.format(k))
                args.append(updates[k])
            query = """
                UPDATE sources
                SET {}
                WHERE source_id=?
                """.format(','.join(query_args))

            args.append(source_id)
            conn.execute(query, args)







