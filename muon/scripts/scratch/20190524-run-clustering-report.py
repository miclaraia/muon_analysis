import random
import numpy as np
import os
from tqdm import tqdm
from pprint import pprint
import logging
import pickle
import click
from sklearn.metrics import f1_score
import sklearn.metrics
from sklearn.metrics import homogeneity_score
from pandas import DataFrame

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


def _gen_splits(label_name):
    query = """
        SELECT S.charge, S.split_id, L.label
        FROM subjects AS S
        INNER JOIN subject_labels as L ON
            S.subject_id=L.subject_id
        INNER JOIN image_subjects as I ON
            S.subject_id=I.subject_id
        INNER JOIN images AS IM ON
            IM.image_id=I.image_id
        WHERE IM.group_id in (10,11,12,13)
            AND L.label_name=%s
    """
    database = Database()
    with database.conn as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (label_name,))
            data = {0: [], 1: []}
            for row in tqdm(cursor):
                charge = np.frombuffer(row[0], dtype=np.float32)
                split = row[1]
                label = row[2]

                data[split].append([charge, label])

    for i in data:
        data[i] = list(zip(*data[i]))
        data[i] = np.array(data[i][0]), np.array(data[i][1])
    td = (np.zeros((0, 499)), np.zeros((0,)))

    splits = {'train': data[0], 'test': data[1], 'train_dev': td}
    return splits


@cli.command()
@click.argument('splits_path')
@click.argument('label_name')
def gen_splits(splits_path, label_name):
    splits = _gen_splits(label_name)
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
    pprint(report)

    split = '_'.join(splits_path.split('/')[-1].split('.')[0].split('_')[1:])
    os.rename(os.path.join(save_dir, 'report.pkl'),
              os.path.join(save_dir, 'report_{}.pkl'.format(split)))
    os.rename(os.path.join(save_dir, 'pca_plot.png'),
              os.path.join(save_dir, 'pca_plot_{}.png'.format(split)))

    from pandas import DataFrame
    cmap = DataFrame(list(zip(*report['cmap'])))
    print(cmap)
    print(report['metrics'])

    # print(DataFrame(list(zip(*dec.get_cluster_map(*splits['test'])))))

    import code
    code.interact(local={**globals(), **locals()})


@cli.command()
@click.argument('splits_path')
@click.option('--sample_size', default=100, type=int)
def random_assignment(splits_path, sample_size):
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)

    config = Config.load(save_dir)

    x, y = splits['test']

    trials = []
    for _ in range(5):
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

        clicks = EfficiencyStudy._simulate(
            y, cluster_pred,
            config.n_classes,
            config.n_clusters,
            sample_size=sample_size)

        trials.append((f1c, h, nmi, clicks))
        logger.info(trials[-1])

    trials = DataFrame(trials)
    print(trials)
    print(trials.mean())


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
