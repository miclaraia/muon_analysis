import os
import random
import numpy as np
import click
import json
import pandas

from muon.subjects.storage import Storage


def get_labels(storage, label_name, rotation):
    subject_ids = storage.labeled_subjects(label_name)
    if not rotation:
        return subject_ids
    else:
        return ['{}_{}'.format(s, n) for s in subject_ids for n in range(6)]


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


@click.group(invoke_without_command=True)
@click.option('--subject_data', required=True)
@click.option('--train_labels', required=True)
@click.option('--true_labels', required=True)
@click.option('--output_file', required=True)
@click.option('--train_rotation', is_flag=True)
@click.option('--true_rotation', is_flag=True)
def main(subject_data, train_labels, true_labels, output_file,
         train_rotation, true_rotation):
    storage = Storage(subject_data)
    label_keys = {'train': train_labels, 'true': true_labels}

    train_labels = set(get_labels(storage, label_keys['train'], train_rotation))
    true_labels = set(get_labels(storage, label_keys['true'], true_rotation))
    all_labels = list(train_labels | true_labels)

    train_labels = train_labels - true_labels

    train_splits = {
        'train': .75,
        'train_dev': .25
    }
    test_splits = {
        'valid': .75,
        'test': .25
    }

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

    with open(output_file, 'w') as file:
        json.dump(splits, file)


main()
