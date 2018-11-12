import os
import csv
from tqdm import tqdm
import click
import json

from muon.subjects.storage import Storage
from muon.project.hdf_images import HDFImages

agg_data_f = os.path.join(
    os.environ.get('MUOND'),
    'zooniverse', 'MH2', 'image_structure', 'production_data.h5')


@click.group(invoke_without_command=True)
@click.argument('name')
@click.argument('labels_csv')
@click.argument('subjects_h5')
def main(name, labels_csv, subjects_h5):
    labels = []
    print('Starting')
    with open(labels_csv, 'r') as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader):
            labels.append((row['subject'], int(row['label'])))

    storage = Storage(subjects_h5)
    if 'labels' not in storage._file.attrs:
        storage._file.attrs['labels'] = json.dumps({})
    print(len(storage._file['subjects']))
    skipped = storage.add_labels(name, labels)

    print('Skipped {} subjects'.format(len(skipped)))


if __name__ == '__main__':
    main()






