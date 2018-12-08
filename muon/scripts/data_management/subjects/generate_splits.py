import os
import random
import numpy as np
import click
import json
import pandas
import pickle

from muon.subjects.storage import Storage


def get_labels(storage, label_name, rotation):
    subject_ids = storage.labeled_subjects(label_name)
    if rotation:
        return ['{}_{}'.format(s, n) for s in subject_ids for n in range(6)]
    else:
        return subject_ids


def make_splits(keys, splits):
    keys = list(keys)
    splits = splits.copy()
    keys = np.random.permutation(keys)
    l = len(keys)

    for split in splits:
        n = np.round(l * splits[split], 0).astype(np.int)
        splits[split] = [str(s) for s in keys[:n]]
        keys = keys[n:]

    print({s: len(splits[s]) for s in splits})
    print('subjects assigned: {}'.format(
        sum([len(v) for v in splits.values()])))
    print('unassigned subjects: {}'.format(len(keys)))
    return splits


def check_split_fraction(splits):
    s = sum(splits.values())
    if s != 1:
        print(splits)
        raise ValueError('Sum of fractions is not 1:', s)
    return True


@click.group(invoke_without_command=True)
@click.option('--subject_data', required=True)
@click.option('--train_label_name', required=True)
@click.option('--true_label_name', required=True)
@click.option('--splits_out', required=True)
@click.option('--xy_out', required=True)
@click.option('--train', type=float, required=True)
@click.option('--train_dev', type=float, required=True)
@click.option('--valid', type=float, required=True)
@click.option('--test', type=float, required=True)
@click.option('--train_rotation', is_flag=True)
@click.option('--true_rotation', is_flag=True)
def main(subject_data, train_label_name, true_label_name, splits_out, xy_out,
         train, train_dev, valid, test, train_rotation, true_rotation):
    storage = Storage(subject_data)
    label_keys = {'train': train_label_name, 'true': true_label_name}

    train_labels = set(get_labels(storage, label_keys['train'], train_rotation))
    true_labels = set(get_labels(storage, label_keys['true'], true_rotation))
    all_labels = list(train_labels | true_labels)

    train_labels = train_labels - true_labels

    train_splits = {
        'train': train,
        'train_dev': train_dev
    }
    test_splits = {
        'valid': valid,
        'test': test
    }

    check_split_fraction(train_splits)
    check_split_fraction(test_splits)

    splits = {}
    splits.update(make_splits(train_labels, train_splits))
    splits.update(make_splits(true_labels, test_splits))

    print('True labeled subjects: {}'.format(len(true_labels)))
    print('Training labeled subjects: {}'.format(len(train_labels)))

    stats = np.zeros((2, 4))
    keys = ['train', 'train_dev', 'valid', 'test']
    stats[0,:] = [len(splits[s]) for s in keys]
    stats[1,:] = stats[0,:] / sum(stats[0,:])
    df = pandas.DataFrame(stats, columns=keys)
    df.insert(0, '', ['n', 'fraction'])
    print(df)

    with open(splits_out, 'w') as file:
        json.dump(splits, file)

    # Generate xy data
    l = {k: len(splits[k]) for k in splits}
    print(l)

    labels = list(set([s.split('_')[0] for s in all_labels]))
    subjects = storage.get_subjects(labels)
    storage.close()
    for k in train_splits:
        print(k)
        splits[k] = subjects.get_xy(splits[k], train_label_name)
        print(splits[k][0].shape)
    for k in test_splits:
        print(k)
        splits[k] = subjects.get_xy(splits[k], true_label_name)
        print(splits[k][0].shape)

    l2 = {k: splits[k][0].shape[0] for k in splits}
    print(l2)
    assert l == l2

    with open(xy_out, 'wb') as f:
        pickle.dump(splits, f)


main()
