import json
import click
import pickle
import numpy as np

from muon.subjects.storage import Storage


@click.group(invoke_without_command=True)
@click.argument('splits_file')
@click.argument('subject_data')
@click.argument('labels')
@click.argument('output')
@click.option('--rotation', is_flag=True)
def main(splits_file, subject_data, labels, output, rotation):
    storage = Storage(subject_data)
    subjects = storage.get_subjects(storage.labeled_subjects(labels))
    with open(splits_file, 'r') as f:
        splits = json.load(f)

    for k in splits:
        print(k)
        splits[k] = subjects.subset(splits[k]).get_xy(rotation, labels)
        print(splits[k][0].shape)

    with open(output, 'wb') as f:
        pickle.dump(splits, f)


if __name__ == '__main__':
    main()
