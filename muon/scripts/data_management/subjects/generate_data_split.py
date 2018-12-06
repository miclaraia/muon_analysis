import os
import random
import numpy as np
import click
import json
import pickle

from muon.subjects.storage import Storage


@click.group(invoke_without_command=True)
@click.option('--subject_data', required=True)
@click.option('--label_name', required=True)
@click.option('--splits_out', required=True)
@click.option('--xy_out', required=True)
@click.option('--rotation', is_flag=True)
def main(subject_data, label_name, splits_out, xy_out, rotation):
    storage = Storage(subject_data)
    subjects = storage.get_subjects(storage.labeled_subjects(label_name))
    storage.close()

    keys = [s for s in subjects.keys()]
    if rotation:
        keys = ['{}_{}'.format(s, n) for s in keys for n in range(6)]
    keys = np.random.permutation(keys)

    splits = {
        'test': .25,
        'train': .4,
        'train_dev': .175,
        'valid': .175
    }
    print('Sum of percentages: ', sum(splits.values()))

    l = len(keys)
    S = np.array(list(splits.keys()))
    N = np.round(np.array([splits[k] for k in S]) * l, 0).astype(np.int)
    print(N, sum(N), l)

    splits = {}
    for i, s in enumerate(S):
        splits[s] = keys[:N[i]]
        keys = keys[N[i]:]

    print(sum([len(splits[s]) for s in splits]))
    splits = {k: list(splits[k]) for k in splits}
        
    with open(splits_out, 'w') as file:
        json.dump(splits, file)

    # Generate xy data
    l = sum([len(splits[k]) for k in splits])
    print([len(splits[s]) for s in splits])
    for k in splits:
        print(k)
        splits[k] = subjects.get_xy(splits[k], label_name)
        print(splits[k][0].shape)
    l2 = sum([splits[k][0].shape[0] for k in splits])
    print(l, l2)
    print([splits[k][0].shape[0] for k in splits])
    assert l == l2

    with open(xy_out, 'wb') as f:
        pickle.dump(splits, f)


main()
