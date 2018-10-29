import os
import random
import numpy as np
import click
import json

from muon.subjects.storage import Storage


@click.group(invoke_without_command=True)
@click.argument('subject_data')
@click.argument('label_name')
@click.argument('output_file')
def main(subject_data, label_name, output_file):
    storage = Storage(subject_data)
    subjects = storage.get_subjects(storage.labeled_subjects(label_name))

    keys = ['{}_{}'.format(s, n) for s in subjects.keys() for n in range(6)]
    keys = np.random.permutation(keys)

    split = {
        'test': .25,
        'train': .4,
        'train_dev': .175,
        'valid': .175
    }
    print('Sum of percentages: ', sum(split.values()))

    l = len(keys)
    S = np.array(list(split.keys()))
    N = np.round(np.array([split[k] for k in S]) * l, 0).astype(np.int)
    print(N, sum(N), l)

    split = {}
    for i, s in enumerate(S):
        split[s] = keys[:N[i]]
        keys = keys[N[i]:]

    print(sum([len(split[s]) for s in split]))
    split = {k: list(split[k]) for k in split}
        
    with open(output_file, 'w') as file:
        json.dump(split, file)



main()
