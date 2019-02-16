import random
import numpy as np
import os
from tqdm import tqdm

from redec_keras.models.decv2 import DECv2, Config

SPLITS = {k: i for i, k in enumerate(['train', 'test', 'valid', 'train_dev'])}


class Clustering:

    @classmethod
    def train_decv2(cls, config, subject_storage):
        database = subject_storage.database
        with database.conn as conn:
            subjects = subject_storage.get_split_subjects('train')
            x = subjects.get_x()

        dec = DECv2(config, x.shape)
        dec.init(x)

        no_shape = (0, x.shape[1])
        dec.clustering(
            (x, None),
            (np.zeros(no_shape), None),
            (np.zeros(no_shape), None))

    @classmethod
    def train_multitask(cls, config, subject_storage):
        pass

    @classmethod
    def train_redec(cls, config, subject_storage):
        pass

    @classmethod
    def assign_clusters(cls, config, subject_storage, cluster_name,
                        batch=None):
        database = subject_storage.database

        models = {
            'decv2': DECv2,
            'multitask': None,
            'redec': None
        }
        if batch:
            subjects = subject_storage.get_subject_batch(batch)
        else:
            subjects = subject_storage.get_all_subjects()
        subject_ids = subjects.keys()
        x = subjects.get_x()

        dec = models[config.type].load(config.save_dir, np.zeros((0, 499)))
        cluster_pred = zip(subject_ids, dec.predict_clusters(x))

        with database.conn as conn:
            for subject_id, cluster in cluster_pred:
                database.Clustering.add_subject_cluster(
                    conn, subject_id, cluster_name, int(cluster))
            conn.commit()

    @classmethod
    def cluster_assignment_counts(cls, subject_storage, label_name):
        database = subject_storage.database

        with database.conn as conn:
            clusters = {}
            for cluster, _, label in tqdm(
                    database.Clustering.get_cluster_labels(conn, label_name)):

                if cluster not in clusters:
                    clusters[cluster] = np.zeros((2,))
                clusters[cluster][label] += 1

            clusters = np.array([clusters[c] for c in sorted(clusters)])

            return clusters

