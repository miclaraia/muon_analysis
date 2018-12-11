import os
import random
import numpy as np
import click
import json
import pandas
import pickle

from muon.subjects.storage import Storage


@click.group(invoke_without_command=True)
@click.option('--subject_file', required=True)
@click.option('--splits_source', required=True)
@click.option('--xy_source', required=True)
@click.option('--xy_out', required=True)
@click.option('--label_name', required=True)
def main(subject_file, splits_source, xy_source, xy_out, label_name):

    with open(splits_source, 'r') as f:
        subject_splits = json.load(f)
    with open(xy_source, 'rb') as f:
        splits = pickle.load(f)

    storage = Storage(subject_file)
    labels = storage.labeled_subjects(label_name)
    subjects = storage.get_subjects(labels)
    for k in ['train', 'train_dev']:
        print(k)
        print(splits[k][0].shape)
        splits[k] = subjects.get_xy(subject_splits[k], label_name)
        print(splits[k][0].shape)

    for k in splits_source:
        print(k)
        assert len(splits_source[k]) == splits[k][0].shape[0]

    with open(xy_out, 'wb') as f:
        pickle.dump(splits, f)


if __name__ == '__main__':
    main()
