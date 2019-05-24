import h5py
import json
import os
import csv
from tqdm import tqdm
import logging

from muon.config import Config
from muon.database.database import Database
from muon.subjects.storage import Storage
from muon.subjects.subject import Subject
from muon.images.image_parent import ImageParent
from muon.images.image_group_parent import ImageGroupParent
from muon.images import TYPES
ImageMeta = ImageParent.ImageMeta

logger = logging.getLogger(__name__)

root = os.getenv('MUOND')
subjects_fname = os.path.join(root, 'subjects/subject_data_v3.hdf5')
splits_fname = os.path.join(root, 'subjects/split_v2_swap_norotation.json')
golds_fname = os.path.join(root, 'zooniverse/MH2/mh2_golds.csv')
image_fname = os.path.join(root, 'zooniverse/MH2/image_structure/production_data.h5')


def load_subject_data(fname, splits):
    logger.info('Loading subjects')
    logger.info(fname)
    with h5py.File(fname, 'r') as f:
        for subject_id in tqdm(f['subjects']):
            item = f['subjects'][subject_id]
            subject_id = int(subject_id)

            charge = item['charge'][:-1]
            metadata = json.loads(item.attrs['metadata'])
            source_id = 'run_{run}_evt_{evt}_tel_{tel}' \
                .format(**metadata)
            source = 'beta-subjects'
            label = json.loads(item.attrs['label'])
            if label is None:
                label = {}

            split_id = get_split_id(subject_id, splits)

            subject = Subject(
                subject_id, charge, source_id=source_id,
                source=source, label=label)
            yield subject, split_id, label


def load_gold_labels(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row['subject'], row['label']


def load_subject_labels(fname):
    with h5py.File(fname, 'r') as f:
        data = json.loads(f.attrs['labels'])

    for label_name in data:
        for subject_id in data[label_name]:
            label = data[label_name][subject_id]
            yield label_name, subject_id, label


def get_split_id(subject_id, splits):
    for k in splits:
        if str(subject_id) in splits[k]:
            if k in Database.Subject._splits:
                return k
            else:
                raise Exception('split name not found')
    print('No split found for {}'.format(subject_id))
    return None


def load_splits(fname):
    logger.info('Loading splits')
    logger.info(fname)
    with open(fname, 'r') as f:
        return json.load(f)


def load_images(fname):
    logger.info('Loading images')
    logger.info(fname)
    with h5py.File(fname, 'r') as f:
        for group in tqdm(f['groups']):
            group_id = f['groups'][group].attrs['group']
            for image in tqdm(f['groups'][group]['images']):
                item = f['groups'][group]['images'][image]
                image_id = int(item.attrs['id'])
                group_id = int(item.attrs['group'])
                metadata = json.loads(item.attrs['metadata'])

                if 'figure' in metadata:
                    image_meta = ImageMeta(
                        dpi=metadata['figure']['dpi'],
                        offset=metadata['figure']['offset'],
                        height=metadata['figure']['height'],
                        width=metadata['figure']['width'],
                        rows=metadata['figure']['rows'],
                        cols=metadata['figure']['cols']).__dict__
                else:
                    image_meta = None
                zoo_id = int(item.attrs['zoo_id'])

                subjects = [int(s) for s in item['subjects'][:]]

                attrs = {
                    'group_id': group_id,
                    'metadata': None,
                    'cluster': metadata['cluster'],
                    'subjects': subjects,
                    'zoo_id': zoo_id,
                    'image_meta': image_meta
                }

                yield ImageParent(image_id, None, attrs)


def load_groups(fname):
    logger.info('Loading groups')
    logger.info(fname)
    with h5py.File(fname, 'r') as f:
        for group in tqdm(f['groups']):
            group_id = int(f['groups'][group].attrs['group'])

            attrs = {
                'group_type': TYPES['grid'],
                'image_size': 100,
                'image_width': 10,
                'description': None,
                'permutations': 1,
                'cluster_name': '',
                'image_count': 0,
                'zoo_subject_set': None,
            }
            group = ImageGroupParent(group_id, None, attrs)
            yield group


def main():
    config = Config.instance()
    config.config['database']['dbname'] = 'muon_beta_data'

    database = Database()
    print(database.__dict__)

    splits = load_splits(splits_fname)
    with database.conn as conn:
        for subject, split_id, labels in \
                load_subject_data(subjects_fname, splits):
            database.Subject.add_subject(conn, subject, 0, split_id)

            for label_name in labels:
                if label_name == 'hugh':
                    continue
                label = labels[label_name]
                database.Subject.add_subject_label(
                    conn, subject.id, label_name, label)

        for subject_id, label in load_gold_labels(golds_fname):
            database.Subject.add_subject_label(
                conn, subject_id, 'hugh', label)

        for image in load_images(image_fname):
            database.Image.add_image(conn, image)

        for image_group in load_groups(image_fname):
            database.ImageGroup.add_group(conn, image_group)

        conn.commit()


if __name__ == '__main__':
    main()
