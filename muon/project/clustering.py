

from redec_keras.models.decv2 import DECv2, Config
import random

SPLITS = {k: i for i, k in enumerate(['train', 'test', 'valid', 'train_dev'])}


class Clustering:

    @classmethod
    def reserve_test_set(cls, subject_storage):
        database = subject_storage.database
        subject_ids = database.Subject.list_labeled_subjects()

        with subject_storage.database.conn as conn:
            for subject_id in subject_ids:
                database.Clustering.set_subject_test_flag(
                    conn, subject_id, (random.random() < 0.25))

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

        y_pred = dec.clustering(
            (x, None),
            (None, None),
            (None, None))

    @classmethod
    def assign_clusters(cls, config, subject_storage):
        database = subject_storage.database

        models = {
            'dec': DECv2,
            'multitask': None,
            'redec': None
        }
        subjects = subject_storage.get_all_subjects()
        x = subjects.get_x()

        dec = models[config.type].load(config.save_dir, (None, 499))
        cluster_pred = zip(subject_ids, dec.predict_clusters(x))

        with database.conn as conn:
            for subject_id, cluster in cluster_pred:
                database.Clustering.set_subject_cluster(conn, subject_id, cluster)
            conn.commit()


        #dec.report_run(splits)
