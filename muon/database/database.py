import psycopg2
import os
import numpy as np
import json
from tqdm import tqdm
import logging
import decimal

from muon.subjects.subject import Subject
from muon.config import Config

logger = logging.getLogger(__name__)


class Database:

    def __init__(self, host=None, dbname=None, username=None, password=None):
        config = Config.instance().database

        db_kwargs = {
            'host': config.host,
            'database': config.dbname,
            'username': config.username,
            'password': config.password
        }

        db_update = {
            'host': host,
            'database': dbname,
            'username': username,
            'password': password
        }
        for k, v in db_update.items():
            if v is not None:
                db_kwargs[k] = v

        self.db_kwargs = db_kwargs

    @property
    def conn(self):
        return self.__class__.Connection(**self.db_kwargs)

    def create_db(self):
        fname = os.path.dirname(__file__)
        fname = os.path.join(fname, 'schema-pg.sql')
        with open(fname, 'r') as f:
            query = f.read()

        logger.info('Running db schema')
        queries = query.split(';')

        with self.conn as conn:
            with conn.cursor() as cur:
                cur.execute(query)

            conn.commit()

    class Connection:

        def __init__(self, host, database, username, password):
            self.host = host
            self.database = database
            self.username = username
            self.password = password

            self.conn = None

        def __enter__(self):
            self.conn = psycopg2.connect(
                host=self.host,
                user=self.username,
                password=self.password,
                database=self.database)

            return self.conn

        def __exit__(self, type, value, traceback):
            self.conn.close()


    class Subject:
        _splits = {k: i for i, k in
                   enumerate(['train', 'test', 'valid', 'train_dev', None])}

        @classmethod
        def next_batch(cls, conn):
            with conn.cursor() as cursor:
                cursor.execute('SELECT MAX(batch_id) FROM subjects')
                last_id = cursor.fetchone()[0]
                if last_id is None:
                    return 0
                return last_id + 1

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
                    ','.join(keys), ','.join(['%s' for _ in range(len(keys))]))

            with conn.cursor() as cursor:
                cursor.execute(query, values)

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
                    ','.join(keys), ','.join(['%s' for _ in range(len(keys))]))

            with conn.cursor() as cursor:
                cursor.execute(query, values)


        ### Getting Subjects #########################

        @classmethod
        def get_subject(cls, conn, subject_id):
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT subject_id, charge FROM subjects
                    WHERE subject_id=%s""", (subject_id,))

                row = cursor.fetchone()
            return Subject(
                id=row[0],
                charge=np.frombuffer(row[1], dtype=np.float32)
            )

        @classmethod
        def get_subject_ids_in_batch(cls, conn, batch_id):
            with conn.cursor() as cursor:
                query = """
                    SELECT subject_id FROM subjects WHERE batch_id=%s
                """
                cursor.execute(query, (batch_id,))

                for row in cursor:
                    yield row[0]

        @classmethod
        def get_subject_batch(cls, conn, batch_id):
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT subject_id, charge FROM subjects
                    WHERE batch_id=%s
                    """, (batch_id,))

                for row in cursor:
                    yield Subject(
                        id=row[0],
                        charge=np.frombuffer(row[1], dtype=np.float32))

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
        def get_image_subjects(cls, conn, image_id):
            query = """
                SELECT S.subject_id, S.charge
                FROM subjects as S
                INNER JOIN image_subjects as I
                    ON I.subject_id=S.subject_id
                WHERE I.image_id=%s
            """
            with conn.cursor() as cursor:
                cursor.execute(query, (image_id,))

                row = cursor.fetchone()
            return Subject(
                id=row[0],
                charge=np.frombuffer(row[1], dtype=np.float32)
            )

        @classmethod
        def get_all_subjects(cls, conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT subject_id, charge FROM subjects')
                for row in cursor:
                    yield Subject(
                        id=row[0],
                        charge=np.frombuffer(row[1], dtype=np.float32)
                    )

        @classmethod
        def get_subject_label(cls, conn, subject_id, label_name):
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT label FROM subject_labels WHERE
                    subject_id=%s AND label_name=%s""", (subject_id, label_name))
                return cursor.fetchone()[0]

        @classmethod
        def get_source_subject(cls, conn, source_id):
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT subject_id FROM subjects
                    WHERE source_id=%s""", (source_id,))
                row = cursor.fetchone()
                if row:
                    return row[0]

        @classmethod
        def list_label_names(cls, conn):
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT label_name FROM subject_labels GROUP BY label_name')
                return [row[0] for row in cursor]

        @classmethod
        def list_labeled_subjects(self, conn, label_name):
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT subject_id FROM subject_labels WHERE label_name=%s',
                    label_name)
                return [row[0] for row in cursor]

        @classmethod
        def list_subjects(self, conn):
            with conn.cursor() as cursor:
                cursor.execute('SELECT subject_id FROM subjects')
                for row in cursor:
                    yield row[0]

        @classmethod
        def set_subject_split(cls, conn, subject_id, split_name):
            with conn.cursor() as cursor:
                split_id = cls._splits[split_name]
                cursor.execute("""
                    UPDATE subjects SET split_id=%s
                    WHERE subject_id=%s""", (split_id, subject_id)
                    )

        @classmethod
        def get_split_subjects(cls, conn, split_name):
            split_id = cls._splits[split_name]
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT subject_id FROM subjects WHERE split_id=%s',
                    (split_id,))
                for row in cursor:
                    yield row[0]

        @classmethod
        def get_split_subjects_batch(cls, conn, split_name, batch_id):
            split_id = cls._splits[split_name]
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT subject_id FROM subjects
                    WHERE split_id=%s and batch_id=%s
                    """, (split_id, batch_id))
                for row in cursor:
                    yield row[0]

        @classmethod
        def get_subjects_by_split_group(
                cls, conn, split_name, image_groups):
            query = """
                SELECT S.subject_id, S.charge, L.label FROM subjects
                INNER JOIN image_subjects AS SI
                    ON SI.subject_id=S.subject_id
                INNER JOIN images AS I
                    ON I.image_id=SI.image_id
                WHERE I.group_id IN %s
                    AND S.split_id=%s
            """

            split_id = cls._splits[split_name]
            args = (tuple(image_groups), split_id)
            with conn.cursor() as cursor:
                cursor.execute(query, args)
                for row in cursor:
                    yield Subject(
                        id=row[0],
                        charge=np.frombuffer(row[1], dtype=np.float32))
        @classmethod
        def get_subjects_labels_by_split_group(
                cls, conn, split_name, label_name, image_groups):
            query = """
                SELECT S.subject_id, S.charge, L.label FROM subjects
                INNER JOIN image_subjects AS SI
                    ON SI.subject_id=S.subject_id
                INNER JOIN images AS I
                    ON I.image_id=SI.image_id
                INNER JOIN subject_labels AS L
                    ON L.subject_id=S.subject_id
                WHERE I.group_id IN %s
                    AND S.split_id=%s
                    AND L.label_name=%s
            """

            split_id = cls._splits[split_name]
            args = (tuple(image_groups), split_id, label_name)
            with conn.cursor() as cursor:
                cursor.execute(query, args)
                for row in cursor:
                    yield Subject(
                        id=row[0],
                        charge=np.frombuffer(row[1], dtype=np.float32),
                        label=row[2])

        @classmethod
        def get_split_subjects_batches(cls, conn, split_name, batches):
            for batch in batches:
                for row in cls.get_split_subjects_batch(
                        conn, split_name, batch):
                    yield row

    class Clustering:

        @classmethod
        def add_subject_cluster(cls, conn, subject_id, cluster_name, cluster):
            with conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM subject_clusters
                    WHERE subject_id=%s AND cluster_name=%s""",
                    (subject_id, cluster_name))

                cursor.execute("""
                    INSERT INTO subject_clusters
                    (subject_id, cluster_name, cluster)
                    VALUES (%s,%s,%s)""", (subject_id, cluster_name, cluster))

        @classmethod
        def get_cluster_assignments(cls, conn, cluster_name, batch=None):
            with conn.cursor() as cursor:
                if batch:
                    cursor.execute("""
                        SELECT subject_clusters.subject_id,
                               subject_clusters.cluster
                        FROM subject_clusters
                        INNER JOIN subjects
                            ON subject_clusters.subject_id=subjects.subject_id
                        WHERE
                            cluster_name=%s
                            AND subjects.batch_id=%s
                        """, (cluster_name, batch))
                else:
                    cursor.execute("""
                        SELECT subject_id, cluster FROM subject_clusters
                        WHERE cluster_name=%s""", (cluster_name,))

                clusters = {}
                for subject_id, cluster in tqdm(cursor):
                    if cluster not in clusters:
                        clusters[cluster] = []
                    clusters[cluster].append(subject_id)
                return clusters

        @classmethod
        def get_cluster_labels(cls, conn, cluster_name, label_name):
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT subject_clusters.cluster, 
                        subject_clusters.subject_id, subject_labels.label
                        FROM subject_clusters
                    INNER JOIN subject_labels
                        ON subject_clusters.subject_id=subject_labels.subject_id
                    WHERE subject_clusters.cluster_name=%s 
                        AND subject_labels.label_name=%s
                    ORDER BY subject_clusters.cluster ASC""",
                    (cluster_name, label_name,))

                for row in cursor:
                    yield row

        @classmethod
        def get_cluster_subjects(cls, conn, cluster_name, cluster):
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT subject_id FROM subject_clusters
                    WHERE cluster_name=%s AND cluster=%s""",
                    (cluster_name, cluster))

                for row in cursor:
                    yield row[0]

    class Image:

        @classmethod
        def next_id(cls, conn):
            with conn.cursor() as cursor:
                cursor.execute('SELECT MAX(image_id) FROM images')
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
                    ','.join(keys), ','.join(['%s' for _ in range(len(keys))]))

            with conn.cursor() as cursor:
                cursor.execute(query, values)

                for i, subject_id in enumerate(image.subjects):
                    cursor.execute("""
                        INSERT INTO image_subjects
                        (subject_id, image_id, group_id, image_index)
                        VALUES (%s,%s,%s,%s)""",
                        (subject_id, image.image_id, image.group_id, i))

            return image.image_id

        @classmethod
        def update_image(cls, conn, image_id, updates):
            query_args = []
            args = []
            for k in updates:
                query_args.append('{}=%s'.format(k))
                args.append(updates[k])
            query = """
                UPDATE images
                SET {}
                WHERE image_id=%s
                """.format(','.join(query_args))

            args.append(image_id)
            with conn.cursor() as cursor:
                cursor.execute(query, args)

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

            image_attrs = {}
            fig_attrs = {}
            for field in kwargs:
                if 'fig_' in field:
                    item = kwargs[field]
                    if type(item) is decimal.Decimal:
                        item = float(item)
                    fig_attrs[field[4:]] = item
                # elif field == 'metadata':
                    # image_attrs[field] = json.loads(kwargs[field])
                else:
                    image_attrs[field] = kwargs[field]

            image_attrs['image_meta'] = fig_attrs

            return image_attrs

        @classmethod
        def _build_get_query(cls, where=None, exclude_zoo=False,
                             shuffle=False):
            if where is None:
                where = 'image_id'
            if where == 'image_id':
                where = 'WHERE images.image_id=%s'
            elif where == 'group_id':
                where = 'WHERE images.group_id=%s'
                if exclude_zoo:
                    where += ' AND images.zoo_id IS NULL'

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
                'ARRAY_AGG(image_subjects.subject_id)::text[]'
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
            with conn.cursor() as cursor:
                cursor.executemany(query, image_ids)

                for row in cursor:
                    yield cls._parse_image_row(row)

        @classmethod
        def get_image(cls, conn, image_id):
            query = cls._build_get_query(where='image_id')
            with conn.cursor() as cursor:
                cursor.execute(query, (image_id,))

                row = cursor.fetchone()
                return cls._parse_image_row(row)

        @classmethod
        def get_group_images(cls, conn, group_id,
                             exclude_zoo=False, shuffle=False):
            query = cls._build_get_query(
                where='group_id', exclude_zoo=exclude_zoo)

            with conn.cursor() as cursor:
                cursor.execute(query, (group_id,))
                for row in cursor:
                    yield cls._parse_image_row(row)

        @classmethod
        def get_group_images_batched(
                cls, conn, group_id, batch_size,
                exclude_zoo=False, shuffle=False):

            image_ids = cls.get_group_image_ids(
                conn, group_id,
                exclude_zoo=exclude_zoo,
                shuffle=shuffle)
            image_ids = list(image_ids)

            query = cls._build_get_query(
                where='image_id', exclude_zoo=exclude_zoo)
            for batch in np.array_split(np.array(image_ids), batch_size):
                query_sub = '({})'.format(
                    ','.join(['%s' for _ in range(len(batch))]))
                query_ = query.replace(
                    'image_id=%s', 'image_id IN {}'.format(query_sub))
                logger.debug(query_)

                with conn.cursor() as cursor:
                    cursor.execute(query_, list(batch))
                    for row in cursor:
                        yield cls._parse_image_row(row)

        @classmethod
        def get_group_image_ids(cls, conn, group_id,
                                exclude_zoo=False, shuffle=False):
            query = """
                SELECT image_id FROM images
                WHERE group_id=%s
            """
            if exclude_zoo:
                query += " AND zoo_id IS NULL"
            if shuffle:
                query += " ORDER BY RANDOM()"

            with conn.cursor() as cursor:
                cursor.execute(query, (group_id,))
                for row in cursor:
                    yield row[0]

        @classmethod
        def get_image_subjects(cls, conn, image_id):
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT subject_id FROM image_subjects
                    WHERE image_id=%s
                    ORDER BY image_index ASC""", (image_id,))
                for row in cursor:
                    yield row[0]

    class ImageGroup:
        # TODO add update methods

        @classmethod
        def next_id(cls, conn):
            with conn.cursor() as cursor:
                cursor.execute('SELECT MAX(group_id) FROM image_groups')
                last_id = cursor.fetchone()[0]

            if last_id is None:
                return 10
            return last_id + 1

        @classmethod
        def delete_group(cls, conn, group_id):
            print('Deleting group:', group_id, type(group_id))
            with conn.cursor() as cursor:
                cursor.execute('DELETE FROM image_groups WHERE group_id=%s', (group_id,))
                cursor.execute('DELETE FROM images WHERE group_id=%s', (group_id,))
                cursor.execute('DELETE FROM image_subjects WHERE group_id=%s', (group_id,))

        @classmethod
        def add_group(cls, conn, group):

            data = {
                'group_id': group.group_id,
                'group_type': group.group_type,
                'image_count': group.image_count,
                'cluster_name': group.cluster_name,
                'image_size': group.image_size,
                'image_width': group.image_width,
                'description': group.description,
                'permutations': group.permutations,
                'zoo_subject_set': group.zoo_subject_set,
            }
            keys, values = zip(*data.items())

            query = 'INSERT INTO image_groups ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['%s' for _ in range(len(keys))]))
            logger.debug(query)
            logger.debug(values)

            with conn.cursor() as cursor:
                cursor.execute(query, values)

            return group.group_id

        @classmethod
        def update_group(cls, conn, group_id, updates):
            query_args = []
            args = []
            for k in updates:
                query_args.append('{}=%s'.format(k))
                args.append(updates[k])
            query = """
                UPDATE image_groups
                SET {}
                WHERE group_id=%s
                """.format(','.join(query_args))

            args.append(group_id)
            with conn.cursor() as cursor:
                cursor.execute(query, args)

        @classmethod
        def list_groups(cls, conn):
            with conn.cursor() as cursor:
                cursor.execute('SELECT group_id FROM image_groups')
                return [row[0] for row in cursor]

        @classmethod
        def get_group(cls, conn, group_id):
            fields = [
                'image_count',
                'group_type',
                'cluster_name',
                'image_size',
                'image_width',
                'description',
                'permutations',
                'zoo_subject_set',
            ]

            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT {}
                    FROM image_groups WHERE group_id=%s""".format(','.join(fields)),
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

        @classmethod
        def get_groups_subject_labels(cls, conn, label_name, groups):
            query = """
                SELECT SI.subject_id, L.label
                FROM image_subjects as SI
                INNER JOIN images as I
                    ON I.image_id=SI.image_id
                INNER JOIN subject_labels as L
                    ON L.subject_id=SI.subject_id
                WHERE L.label_name=%s and I.group_id IN %s
            """
            with conn.cursor() as cursor:
                cursor.execute(query, (label_name, groups))
                for row in cursor:
                    yield row

    class Source:

        @classmethod
        def add_source(cls, conn, source):
            data = {
                'source_id': source.source_id,
                'source_type': source.source_type,
                'hash': source.hash,
                'updated': source.updated,
            }

            keys, values = zip(*data.items())

            query = 'INSERT INTO sources ({}) VALUES ({})'.format(
                    ','.join(keys), ','.join(['%s' for _ in range(len(keys))]))

            with conn.cursor() as cursor:
                cursor.execute(query, values)

        @classmethod
        def get_source(cls, conn, source_id):
            fields = ['hash', 'updated', 'source_type']

            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT {}
                    FROM sources WHERE source_id=%s""".format(','.join(fields)),
                    (source_id,))
                row = cursor.fetchone()

            row = {fields[i]: v for i, v in enumerate(row)}
            return row

        @classmethod
        def source_exists(cls, conn, source_id):
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT source_id FROM sources
                    WHERE source_id=%s""", (source_id,))
                return cursor.fetchone() is not None

        @classmethod
        def update_source(cls, conn, source_id, updates):
            query_args = []
            args = []
            for k in updates:
                query_args.append('{}=%s'.format(k))
                args.append(updates[k])
            query = """
                UPDATE sources
                SET {}
                WHERE source_id=%s
                """.format(','.join(query_args))

            args.append(source_id)

            with conn.cursor() as cursor:
                cursor.execute(query, args)

    class ImageWorker:

        @classmethod
        def next_id(cls, conn):
            with conn.cursor() as cursor:
                cursor.execute('SELECT MAX(job_id) FROM workers')
                last_id = cursor.fetchone()[0]
            if last_id is None:
                return 0
            return last_id + 1

        @classmethod
        def add_jobs(cls, conn, jobs):
            query1 = """
                INSERT INTO workers (job_id,job_type,job_status)
                VALUES (%s,%s,%s)
            """
            query2 = """
                INSERT INTO worker_images (job_id,image_id) VALUES (%s,%s)
            """

            print(jobs)

            def job_iter():
                for job in jobs:
                    yield (
                        (job.job_id, job.job_type, 0),
                        [(job.job_id, int(i)) for i in job.image_ids]
                    )
                print(job.__dict__)
                print(next(job_iter()))

            with conn.cursor() as cursor:
                for arg1, arg2 in job_iter():
                    cursor.execute(query1, arg1)
                    print(arg1, arg2)
                    cursor.executemany(query2, arg2)

        @classmethod
        def set_job_status(cls, conn, job_id, job_status):
            query = """
                UPDATE workers
                SET job_status=%s
                WHERE job_id=%s
            """
            with conn.cursor() as cursor:
                cursor.execute(query, (job_status, job_id))

        @classmethod
        def get_job(cls, conn):
            query = """
                SELECT job_id,job_type,job_status
                FROM workers
                WHERE job_status=0
                LIMIT 1
            """
            with conn.cursor() as cursor:
                cursor.execute(query)
                row = cursor.fetchone()

            if row is not None:
                fields = ['job_id', 'job_type', 'job_status']
                row = {f: row[i] for i, f in enumerate(fields)}
                row['image_ids'] = None

            return row

        @classmethod
        def get_job_images(cls, conn, job_id):
            fields = ','.join([
                'images.image_id',
                'images.group_id',
                'images.cluster',
                'images.metadata',
                'images.zoo_id',
                'images.fig_dpi',
                'images.fig_offset',
                'images.fig_height',
                'images.fig_width',
                'images.fig_rows',
                'images.fig_cols',
                't2.subject_ids',
                'image_groups.image_width',
                'image_groups.group_type',
                ])

            query = """
                SELECT {fields}
                FROM workers
                INNER JOIN worker_images
                    ON workers.job_id=worker_images.job_id
                INNER JOIN (
                    SELECT image_id, ARRAY_AGG(subject_id)::text[]
                        as subject_ids
                    FROM image_subjects
                    GROUP BY image_id
                ) AS t2 ON t2.image_id=worker_images.image_id
                INNER JOIN images
                    ON images.image_id=worker_images.image_id
                INNER JOIN image_subjects
                    ON image_subjects.image_id=worker_images.image_id
                INNER JOIN image_groups
                    ON image_groups.group_id=images.group_id
                WHERE workers.job_id=%s
            """.format(fields=fields)

            logger.info(query)
            with conn.cursor() as cursor:
                cursor.execute(query, (job_id,))

                for row in cursor:
                    logger.debug(row)
                    logger.debug(row[:-3])
                    logger.debug(type(row[:-3]))

                    # row = list(row)
                    # row[11] = row[11][1:-1].split(',')
                    image = Database.Image._parse_image_row(row[:-2])
                    image_width = row[-2]
                    image_type = row[-1]
                    yield image, image_width, image_type

        @classmethod
        def clear_jobs(cls, conn):
            with conn.cursor() as cursor:
                cursor.execute('DELETE FROM workers')
                cursor.execute('DELETE FROM worker_images')
