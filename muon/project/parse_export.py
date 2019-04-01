import csv
import json
from datetime import datetime
import os
from tqdm import tqdm
import importlib
import logging

from muon.images.image import Image
from muon.images.image_group import ImageGroup

logger = logging.getLogger(__name__)


# class Config:
# 
#     def __init__(self, image_groups, launch_date, **kwargs):
#         self.tool_name = kwargs.get('tool_name', 'Tool name')
#         self.task_A = kwargs.get('task_A', 'T0')
#         self.task_B = kwargs.get('task_B', ['T1', 'T3'])
#         self.launch_date = launch_date
#         self.time_format = '%Y-%m-%d %H:%M:%S %Z'
#         self.image_groups = image_groups
#         self.image_dir = kwargs.get('image_dir')
# 
#         task_map = {
#             'all_muons': ['All Muons'],
#             'most_muons': ['Majority are Muons', 'No clear majority'],
#             'most_nonmuons': ['Minority are Muons'],
#             'no_muons': ['No Muons']
#         }
#         self.task_map = kwargs.get('task_map', task_map)
# 
#         subject_path = kwargs.get('subject_path')
#         if subject_path:
#             subject_path = os.path.abspath(subject_path)
#         self.subject_path = subject_path
# 
#     def _launch_date(self):
#         fmt = '%Y-%m-%d'
#         return datetime.strptime(self.launch_date, fmt)


# class Clusters:

    # def __init__(self, name, cluster, agg=None, data=None):
        # self.name = name
        # self.cluster = cluster

        # if data is None:
            # data = {}
        # self.data = data

        # self.agg = agg

    # def __call__(self):
        # agg = self.agg
        # if agg is None:
            # agg = Aggregate.load(self.name)

        # labels = agg.subject_labels()
        # self.cluster.predict_labels(labels)


class Aggregate:
    def __init__(self, database, config):
        self.database = database
        self.config = config

        self.images = {}
        self.subjects = {}

    def aggregate(self, fname):

        parser = Parse(self.config, self.database)

        for image, n_total, choice, coords in parser.parse(fname):
            image_id = image.image_id

            if image_id not in self.images:
                self.images[image_id] = []

            if choice == 'all_muons':
                self._annotate_image(image_id, n_total, n_total)
                for subject_id in image.subjects:
                    self._annotate_subject(subject_id, 1)

            elif choice == 'no_muons':
                self._annotate_image(image_id, 0, n_total)
                for subject_id in image.subjects:
                    self._annotate_subject(subject_id, 0)

            else:
                clicked_subjects = \
                        list(set(self._clicked_subjects(image, coords)))
                assert len(clicked_subjects) == len(set(clicked_subjects))
                n_clicks = len(clicked_subjects)

                if choice == 'most_muons':
                    v = (0, 1)
                    self._annotate_image(image_id, n_total-n_clicks, n_total)
                elif choice == 'most_nonmuons':
                    v = (1, 0)
                    self._annotate_image(image_id, n_clicks, n_total)
                else:
                    print(choice, image)
                    raise Exception

                for subject_id in image.subjects:
                    if subject_id in clicked_subjects:
                        self._annotate_subject(subject_id, v[0])
                    else:
                        self._annotate_subject(subject_id, v[1])

    def _annotate_image(self, image_id, n_muons, n_total):
        if image_id not in self.images:
            self.images[image_id] = []
        self.images[image_id].append((n_muons, n_total, n_muons/n_total))

    def _annotate_subject(self, subject_id, annotation):
        if subject_id not in self.subjects:
            self.subjects[subject_id] = []

        self.subjects[subject_id].append(annotation)

    def _clicked_subjects(self, image, clicks):
        for x, y in clicks:
            yield image.at_location(x, y)

    def reduce_images(self):
        for image_id in sorted(self.images):
            n, t, _ = zip(*self.images[image_id])
            yield image_id, sum(n) / sum(t)

    def reduce_subjects(self):
        for subject_id in sorted(self.subjects):
            v = self.subjects[subject_id]
            yield subject_id, sum(v)/len(v)

    def dump_images(self, fname):
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'fraction'])

            for image_id, fraction in self.reduce_images():
                writer.writerow([image_id, fraction])

    def dump_subjects(self, fname):
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['subject_id', 'fraction'])

            for image_id, fraction in self.reduce_images():
                writer.writerow([image_id, fraction])




# class Aggregate:
# 
#     def __init__(self, name, config, images=None, data=None, parsed=None):
#         self.name = name
#         self.config = config
# 
#         if images is None:
#             images = MultiGroupImages(config.image_groups)
#         self.images = images
# 
#         if data is None:
#             data = {}
#         self.data = data
#         self.parsed = parsed
# 
#     @classmethod
#     def from_parse(cls, parse):
#         return cls(parse.name, parse.config, parse.images, parsed=parse)
# 
#     def __call__(self):
#         parsed = self.parsed
#         if self.parsed is None:
#             parsed = Parse.load(self.name)
# 
#         images = {}
#         subjects = {}
#         for zoo_id, item in parsed.data.items():
#             image = self.images.get_zoo(zoo_id)
# 
#             for value, clicks in item:
#                 if value == 'all_muons':
#                     l = len(image.subjects)
#                     self._annotate_i(images, image.id, l, l)
# 
#                     for subject in image.subjects:
#                         self._annotate_s(subjects, subject, 1)
# 
#                 elif value == 'no_muons':
#                     l = len(image.subjects)
#                     self._annotate_i(images, image.id, 0, l)
# 
#                     for subject in image.subjects:
#                         self._annotate_s(subjects, subject, 0)
# 
#                 else:
#                     _subjects = self.parse_subjects(image, clicks)
#                     total = len(image.subjects)
#                     l = len(clicks)
# 
#                     if value == 'most_muons':
#                         v = (0, 1)
#                         self._annotate_i(images, image.id, total-l, total)
#                     elif value == 'most_nonmuons':
#                         v = (1, 0)
#                         self._annotate_i(images, image.id, l, total)
# 
#                     for subject in image.subjects:
#                         if subject in _subjects:
#                             self._annotate_s(subjects, subject, v[0])
#                         else:
#                             self._annotate_s(subjects, subject, v[1])
# 
#         data = {'images': images, 'subjects': subjects}
#         self.data = data
#         return data
# 
#     def reduce(self):
#         images = {}
#         for i, v in self.data['images'].items():
#             n, t, _ = zip(*v)
#             images[i] = sum(n) / sum(t)
# 
#         subjects = {}
#         for s, v in self.data['subjects'].items():
#             subjects[s] = sum(v)/len(v)
# 
#         return images, subjects
# 
#     def labeled_subjects(self):
#         return list(self.data['subjects'].keys())
# 
#     def subject_labels(self):
#         _, _subjects = self.reduce()
#         subjects = {}
#         for s, p in _subjects.items():
#             if p >= .5:
#                 subjects[s] = 1
#             else:
#                 subjects[s] = 0
#         return subjects
# 
#     def apply_labels(self, subjects):
#         labels = self.subject_labels()
#         for s, l in labels.items():
#             subjects[s].label = l
# 
#     @staticmethod
#     def _annotate_i(images, image, muons, total):
#         if image not in images:
#             images[image] = []
# 
#         images[image].append((muons, total, muons/total))
# 
#     @staticmethod
#     def _annotate_s(subjects, subject, value):
#         if subject not in subjects:
#             subjects[subject] = []
#         subjects[subject].append(value)
# 
#     def parse_subjects(self, image, coordinates):
#         subjects = []
#         try:
#             for x, y in coordinates:
#                 subjects.append(image.at_location(x, y))
#         except Exception as e:
#             print(image)
#             raise e
#         return subjects
# 
#     @staticmethod
#     def fname(name):
#         return 'agg_dump_%s.json' % name
# 
#     def save(self):
#         fname = self.fname(self.name)
#         fname = muon.data.path(fname)
#         data = {
#             'name': self.name,
#             'config': self.config.__dict__,
#             'data': self.data,
#         }
# 
#         with open(fname, 'w') as file:
#             json.dump(data, file)
# 
#     @classmethod
#     def load(cls, name):
#         fname = cls.fname(name)
#         fname = muon.data.path(fname)
#         with open(fname, 'r') as file:
#             data = json.load(file)
# 
#         config = Config(**data['config'])
# 
#         _data = data['data']
#         _data['images'] = {int(k):v for k, v in _data['images'].items()}
#         _data['subjects'] = {int(k):v for k, v in _data['subjects'].items()}
# 
#         return cls(name, config, data=data['data'])


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
                            image_groups[group_id] = \
                                ImageGroup(group_id, self.database)
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
                print(task)
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


# class Parse:
# 
#     def __init__(self, name, config, data=None):
#         self.name = name
#         self.images = MultiGroupImages(config.image_groups)
#         self.config = config
# 
#         if data is None:
#             data = {}
#         self.data = data
# 
#     def parse(self, fname):
#         with open(fname, 'r') as file:
#             reader = csv.DictReader(file)
# 
#             data = {}
#             for row in reader:
#                 annotations = json.loads(row['annotations'])
#                 image = row['subject_ids']
#                 subject_data = json.loads(row['subject_data'])[image]
# 
#                 if self.should_use(row['created_at'], subject_data):
#                     item = self.parse_annotation(annotations)
# 
#                     if image not in data:
#                         data[image] = []
#                     data[image].append(item)
# 
# 
#         self.data.update(data)
#         return data
# 
#     @staticmethod
#     def fname(name):
#         return 'parse_dump_%s.json' % name
# 
#     def save(self):
#         fname = self.fname(self.name)
#         fname = muon.data.path(fname)
#         data = {
#             'name': self.name,
#             'config': self.config.__dict__,
#             'data': self.data,
#         }
# 
#         with open(fname, 'w') as file:
#             json.dump(data, file)
# 
#     @classmethod
#     def load(cls, name):
#         fname = cls.fname(name)
#         with open(fname, 'r') as file:
#             data = json.load(file)
# 
#         config = Config(**data['config'])
# 
#         return cls(name, config, data=data['data'])
# 
#     def parse_annotation(self, annotations):
#         def choice_map(choice):
#             for k, v in self.config.task_map.items():
#                 if choice in v:
#                     return k
# 
#         choice = 0
#         coords = []
#         for task in annotations:
#             if task['task'] == self.config.task_A:
#                 choice = choice_map(task['value'])
# 
#             elif task['task'] in self.config.task_B:
#                 print(task)
#                 for item in task['value']:
#                     coords.append((item['x'], item['y']))
# 
#         return choice, coords
# 
#     def should_use(self, created_at, subject_data):
#         fmt = self.config.time_format
#         time = datetime.strptime(created_at, fmt)
# 
#         if '#group' in subject_data:
#             group = subject_data['#group']
# 
#             a = time > self.config._launch_date()
#             b = group in self.config.image_groups
# 
#             return a and b
#         return False

