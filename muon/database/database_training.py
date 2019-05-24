import sqlite3
import os
import numpy as np
import json
from tqdm import tqdm
import logging

from muon.subjects.subject import Subject
from muon.config import Config

logger = logging.getLogger(__name__)


class TrainingDatabase:

    def __init__(self, fname):
        if fname is None:
            fname = Config.instance().storage.database
        self.fname = fname

        if not os.path.isfile(fname):
            self.create_db()

    @property
    def conn(self):
        return self.__class__.Connection(self.fname)
        # return sqlite3.connect(self.fname)

    def create_db(self):
        fname = os.path.dirname(__file__)
        fname = os.path.join(fname, 'schema-training.sql')
        with open(fname, 'r') as f:
            query = f.read()

        logger.info('Running db schema')
        queries = query.split(';')

        with self.conn as conn:
            for query in queries:
                if query.startswith('CREATE TABLE'):
                    conn.executescript(query)
            conn.commit()

    def create_indexes(self):
        fname = os.path.dirname(__file__)
        fname = os.path.join(fname, 'schema-training.sql')
        with open(fname, 'r') as f:
            query = f.read()

        logger.info('Running db schema')
        queries = query.split(';')

        with self.conn as conn:
            for query in queries:
                if query.startswith('CREATE INDEX'):
                    conn.executescript(query)
            conn.commit()

    class Connection:

        def __init__(self, fname):
            self.fname = fname
            self.conn = None

        def __enter__(self):
            self.conn = sqlite3.connect(self.fname, timeout=60)
            return self.conn

        def __exit__(self, type, value, traceback):
            self.conn.close()

    @classmethod
    def insert(cls, conn, data):
        query = """
            INSERT INTO subjects (subject_id,charge,label,split_id,rand)
            VALUES (?,?,?,?,RANDOM())
        """
        conn.executemany(query, data)
        conn.commit()

    @classmethod
    def set_cluster(cls, conn, data):
        query = """
            UPDATE subjects SET cluster_id=?
        """
        conn.executemany(query, data)

    @classmethod
    def get_xy(cls, conn):
        query = """
            SELECT subject_id, charge, label
            FROM subjects
            ORDER BY rand
        """
        cursor = conn.execute(query)
        for row in cursor:
            yield (
                row[0],
                np.fromstring(row[1], dtype=np.float32),
                row[2],)



