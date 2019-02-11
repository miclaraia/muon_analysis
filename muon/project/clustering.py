import random
import numpy as np
import os
from tqdm import tqdm

from redec_keras.models.decv2 import DECv2, Config

SPLITS = {k: i for i, k in enumerate(['train', 'test', 'valid', 'train_dev'])}


class Clustering:

    @classmethod
    def reserve_test_set(cls, subject_storage, fraction=0.25):
        # TODO need to make sure previous assignments are preserved
        # or version the test set assignments somehow
        database = subject_storage.database

        with subject_storage.database.conn as conn:
            test_set = list(database.Clustering \
                    .get_subjects(conn, is_test=True))
            train_set = list(database.Clustering \
                    .get_subjects(conn, is_test=False))

            n_add = int(0.25*(len(test_set)+len(train_set)) - len(test_set))
            assert n_add >= 0

            np.random.shuffle(train_set)
            to_add = train_set[:n_add]

            for subject_id in to_add:
                database.Clustering.set_subject_test_flag(
                    conn, subject_id, True)

            conn.commit()

    @classmethod
    def train_decv2(cls, config, subject_storage):
        database = subject_storage.database
        with database.conn as conn:
            subject_ids = database.Clustering.get_subjects(conn, is_test=False)
            subjects = subject_storage.get_subjects(subject_ids)
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
    def assign_clusters(cls, config, subject_storage):
        database = subject_storage.database

        models = {
            'decv2': DECv2,
            'multitask': None,
            'redec': None
        }
        subjects = subject_storage.get_all_subjects()
        subject_ids = subjects.keys()
        x = subjects.get_x()

        dec = models[config.type].load(config.save_dir, np.zeros((0, 499)))
        cluster_pred = zip(subject_ids, dec.predict_clusters(x))

        with database.conn as conn:
            for subject_id, cluster in cluster_pred:
                database.Clustering.set_subject_cluster(conn, subject_id, cluster)
            conn.commit()

    @classmethod
    def assess_clusters(cls, subject_storage):
        database = subject_storage.database

        with database.conn as conn:
            cluster_assignments = database.Clustering.get_cluster_assignments(conn)
            for cluster in cluster_assignments:
                labels = []
                for subject in tqdm(cluster_assignments[cluster]):
                    labels.append(database.Subject \
                        .get_subject_label(conn, subject, 'vegas'))

                yield (cluster, labels)


