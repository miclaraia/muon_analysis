import csv
import json
from datetime import datetime
import os
from tqdm import tqdm
import importlib
import logging
import numpy as np

from muon.images.image import Image
from muon.images.image_group import ImageGroup

logger = logging.getLogger(__name__)


class Aggregate:
    def __init__(self, database, config):
        self.database = database
        self.config = config

        self.images = {}
        self.subjects = {}

    def aggregate(self, fname):

        parser = Parse(self.config, self.database)

        for image, n_total, choice, coords in parser.parse(fname):
            clicked_subjects = self._clicked_subjects(image, coords)
            classification = self._image_classification(
                image, clicked_subjects)

            if choice == 'all_muons':
                classification = np.ones_like(classification)
                self._annotate_image(image, classification)
                self._annotate_subjects(image, classification)

            elif choice == 'no_muons':
                classification = np.zeros_like(classification)
                self._annotate_image(image, classification)
                self._annotate_subjects(image, classification)

            elif choice == 'most_muons':
                classification = 1-classification
                self._annotate_image(image, classification)
                self._annotate_subjects(image, classification)

            elif choice == 'most_nonmuons':
                self._annotate_image(image, classification)
                self._annotate_subjects(image, classification)

            else:
                print(choice, image)
                raise Exception

    def _annotate_image(self, image, classification):
        image_id = image.image_id
        if image_id not in self.images:
            self.images[image_id] = []
        self.images[image_id].append(classification)

    def _annotate_subjects(self, image, classification):
        for i in np.arange(classification.shape[0]):
            subject_id = image.subjects[i]
            annotation = classification[i]
            self._annotate_subject(subject_id, annotation)

    def _annotate_subject(self, subject_id, annotation):
        if subject_id not in self.subjects:
            self.subjects[subject_id] = []

        self.subjects[subject_id].append(annotation)

    def _clicked_subjects(self, image, clicks):
        subjects = []
        for x, y in clicks:
            s = image.at_location(x, y)
            if s is not None:
                subjects.append(s)

        return list(set(subjects))

    def _image_classification(self, image, clicked_subjects):
        data = []
        for subject_id in image.subjects:
            data.append(int(subject_id in clicked_subjects))

        return np.array(data)

    # def reduce_images(self):
        # for image_id in sorted(self.images):
            # n, t, _ = zip(*self.images[image_id])
            # yield image_id, sum(n) / sum(t)

    # def reduce_subjects(self):
        # for subject_id in sorted(self.subjects):
            # v = self.subjects[subject_id]
            # yield subject_id, sum(v)/len(v)

    # def dump_images(self, fname):
        # with open(fname, 'w') as f:
            # writer = csv.writer(f)
            # writer.writerow(['image_id', 'fraction'])

            # for image_id, fraction in self.reduce_images():
                # writer.writerow([image_id, fraction])

    # def dump_subjects(self, fname):
        # with open(fname, 'w') as f:
            # writer = csv.writer(f)
            # writer.writerow(['subject_id', 'fraction'])

            # for image_id, fraction in self.reduce_images():
                # writer.writerow([image_id, fraction])


def load_config(config_path):
    spec = importlib.util.spec_from_file_location('config', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class Parse:

    def __init__(self, database, config):
        self.config = config
        self.database = database

    def parse(self, fname):
        with open(fname, 'r') as file:
            reader = csv.DictReader(file)
            image_groups = {}

            for row in tqdm(reader):
                try:
                    annotations = json.loads(row['annotations'])
                    zoo_id = row['subject_ids']
                    subject_metadata = json.loads(row['subject_data'])[zoo_id]

                    image_id = subject_metadata['id']
                    group_id = subject_metadata['#group']

                    if self.should_use(row['created_at'], subject_metadata):
                        if group_id not in image_groups:
                            logger.info('Loading group %d', group_id)
                            image_groups[group_id] = \
                                ImageGroup.load(group_id, self.database)
                            image_groups[group_id].images.load_all()

                        choice = self.parse_choice(annotations)
                        coords = self.parse_clicks(annotations)
                        image = image_groups[group_id].images[image_id]
                        image_size = image_groups[group_id].image_size

                        yield image, image_size, choice, coords
                except Exception as e:
                    logger.exception(e)
                    print(zoo_id, subject_metadata)
                    continue

    def parse_choice(self, annotations):
        def choice_map(choice):
            for k, v in self.config.task_map.items():
                if choice in v:
                    return k

        for task in annotations:
            if task['task'] == self.config.task_A:
                return choice_map(task['value'])

    def parse_clicks(self, annotations):
        for task in annotations:
            if task['task'] in self.config.task_B:
                # print(task)
                for item in task['value']:
                    yield item['x'], item['y']

    def should_use(self, created_at, subject_metadata):
        fmt = self.config.time_format
        time = datetime.strptime(created_at, fmt)
        launch_date = datetime.strptime(self.config.launch_date, '%Y-%m-%d')

        if '#group' in subject_metadata:
            group = int(subject_metadata['#group'])
            return (time > launch_date) and (group in self.config.image_groups)
        return False
