import random
import numpy as np
import os
from tqdm import tqdm
import logging
import pickle
import click
from sklearn.metrics import f1_score
import sklearn.metrics
from sklearn.metrics import homogeneity_score

from redec_keras.models.decv2 import DECv2, Config
from redec_keras.experiments.simulate_efficiency import EfficiencyStudy
from muon.database.database import Database

from redec_keras.models.utils import get_cluster_to_label_mapping_safe, \
        calc_f1_score, one_percent_fpr

logger = logging.getLogger(__name__)

save_dir = os.path.join(
    os.getenv('MUOND'),
    'clustering_models/aws/project/',
    'model4-8Msubjects-50clusters-b123-2019-02-19T23:27:09-06:00')
# splits_path = os.path.join(save_dir, 'splits_xy2.pkl')


@click.group()
def cli():
    pass

@cli.command()
def counts():
    query = """
        SELECT L.label, count(1)
        FROM subjects AS S
        INNER JOIN subject_labels as L ON
            S.subject_id=L.subject_id
        INNER JOIN image_subjects as I ON
            S.subject_id=I.subject_id
        INNER JOIN images AS IM ON
            IM.image_id=I.image_id
        WHERE S.split_id=1
            AND IM.group_id in (10,11,12,13)
            AND ((L.label_name='vegas_cleaned')
                OR (L.label_name='vegas2' AND L.label=0)
        ) GROUP BY L.label
    """
    database = Database()
    with database.conn as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            for row in cursor:
                print(row)


def _gen_splits(cleaned_only=True, gold_images=True):
    if cleaned_only:
        query = """
            SELECT S.charge, L.label
            FROM subjects AS S
            INNER JOIN subject_labels as L ON
                S.subject_id=L.subject_id
            INNER JOIN image_subjects as I ON
                S.subject_id=I.subject_id
            INNER JOIN images AS IM ON
                IM.image_id=I.image_id
            WHERE S.split_id=1
                AND IM.group_id in (10,11,12,13)
                AND L.label_name='vegas_cleaned'
        """
    elif gold_images:
        query = """
            SELECT S.charge, L.label
            FROM subjects AS S
            INNER JOIN subject_labels as L ON
                S.subject_id=L.subject_id
            INNER JOIN image_subjects as I ON
                S.subject_id=I.subject_id
            INNER JOIN images AS IM ON
                IM.image_id=I.image_id
            WHERE S.split_id=1
                AND IM.group_id in (23,24)
                AND ((L.label_name='vegas_cleaned')
                    OR (L.label_name='vegas2' AND L.label=0)
            )
        """
    else:
        query = """
            SELECT S.charge, L.label
            FROM subjects AS S
            INNER JOIN subject_labels as L ON
                S.subject_id=L.subject_id
            INNER JOIN image_subjects as I ON
                S.subject_id=I.subject_id
            INNER JOIN images AS IM ON
                IM.image_id=I.image_id
            WHERE S.split_id=1
                AND IM.group_id in (10,11,12,13)
                AND ((L.label_name='vegas_cleaned')
                    OR (L.label_name='vegas2' AND L.label=0)
            )
        """
    database = Database()
    with database.conn as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            data = []
            for row in tqdm(cursor):
                charge = np.frombuffer(row[0], dtype=np.float32)
                label = row[1]
                data.append([charge, label])

    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)
    td = (np.zeros((0, 499)), np.zeros((0,)))

    splits = {'train': (x, y), 'test': (x, y), 'train_dev': td}
    return splits


@cli.command()
@click.argument('splits_path')
@click.option('--cleaned_only', is_flag=True)
@click.option('--gold_images', is_flag=True)
def gen_splits(splits_path, cleaned_only, gold_images):
    splits = _gen_splits(cleaned_only=cleaned_only, gold_images=gold_images)
    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)


@cli.command()
@click.argument('splits_path')
def run_report(splits_path):
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    # splits = _gen_splits()
    td = (np.zeros((0, 499)), np.zeros((0,)))
    splits['train_dev'] = td

    dec = DECv2.load(save_dir, np.zeros((0, 499)))
    report = dec.report_run(splits)

    import code
    code.interact(local={**globals(), **locals()})


@cli.command()
@click.argument('splits_path')
def random_assignment(splits_path):
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)

    config = Config.load(save_dir)

    x, y = splits['test']
    _pred = np.random.rand(y.shape[0], 50)

    c_map = get_cluster_to_label_mapping_safe(
        y, _pred.argmax(1), config.n_classes, config.n_clusters,
        toprint=True)[0]

    cluster_pred = _pred.argmax(1)
    # print(cluster_pred, cluster_pred.shape, _pred.shape, y.shape)
    f1c = calc_f1_score(y, cluster_pred, c_map)
    h = homogeneity_score(y, cluster_pred)
    nmi = sklearn.metrics.normalized_mutual_info_score(y, cluster_pred)

    metrics = (None, f1c, h, nmi)
    logger.info(metrics)

    clicks = EfficiencyStudy._simulate_trials(
        y, cluster_pred, 5,
        config.n_classes, config.n_clusters, sample_size=100)
    logger.info(clicks)







@cli.command()
@click.argument('splits_path')
def click_sim(splits_path):
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    x_test, y_test = splits['test']

    dec = DECv2.load(save_dir, np.zeros((0, 499)))
    clicks = EfficiencyStudy.run(dec, x_test, y_test, sample_size=36)
    logger.info(clicks)
    print(clicks)

    # with open(save_dir+'/report.pkl', 'rb') as f:
        # report = pickle.load(f)
    # report['clicks'] = clicks
    # with open(save_dir+'/report.pkl', 'wb') as f:
        # report = pickle.dump(report, f)


if __name__ == '__main__':
    cli()
