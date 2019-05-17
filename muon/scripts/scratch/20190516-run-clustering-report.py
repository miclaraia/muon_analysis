import random
import numpy as np
import os
from tqdm import tqdm
import logging
import pickle
import click

from redec_keras.models.decv2 import DECv2, Config
from muon.database.database import Database

save_dir = os.path.join(
    os.getenv('MUOND'),
    'clustering_models/aws/project/',
    'model4-8Msubjects-50clusters-b123-2019-02-19T23:27:09-06:00')
splits_path = os.path.join(save_dir, 'splits_xy.pkl')


@click.group()
def cli():
    pass


@cli.command()
def gen_splits():
    query = """
        SELECT S.charge, L.label
        FROM subjects AS S
        INNER JOIN subject_labels as L ON
            S.subject_id=L.subject_id
        WHERE L.label_name="vegas_cleaned"
            AND S.split_id=1
    """
    database = Database()
    with database.conn as conn:
        cursor = conn.execute(query)
        data = []
        for row in tqdm(cursor):
            charge = np.fromstring(row[0], dtype=np.float32)
            label = row[1]
            data.append([charge, label])

    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)
    td = (np.zeros((0, 499)), np.zeros((0,)))

    splits = {'train': (x, y), 'test': (x, y), 'train_dev': td}

    with open(splits_path, 'wb') as f:
        pickle.dump(splits, f)


@cli.command()
def run_report():
    dec = DECv2.load(save_dir, np.zeros((0, 499)))

    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
        td = (np.zeros((0, 499)), np.zeros((0,)))
        splits['train_dev'] = td
    dec.report_run(splits)


if __name__ == '__main__':
    cli()
