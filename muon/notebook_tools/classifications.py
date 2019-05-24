import numpy as np
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.metrics
from pandas import DataFrame
import logging
from IPython.display import display

from muon.database.database import Database
from muon.project.parse_export import Aggregate, load_config

logger = logging.getLogger(__name__)


class AggregationAnalysis:

    def __init__(self, agg_dump, cleaned=True):
        with open(agg_dump, 'rb') as f:
            data = pickle.load(f)

        agg = Aggregate(None, None)
        agg.images = data['images']
        agg.subjects = data['subjects']
        agg.swap = data['swap']

        self.agg = agg
        self._labels = None
        self._majority = None
        self._swap = None
        self._first = None
        self._np_labels = None

        self._subjects = None

        self._metrics = None
        self._cleaned = cleaned

    def summary(self):
        agg = self.agg
        n_subjects = np.sum([len(agg.subjects[s]) for s in agg.subjects])
        n_grids = np.sum([len(agg.images[s]) for s in agg.images])
        metrics = dict(self.flatten_metrics)
        print(metrics)
        print('Measuring on {} subjects'.format(len(self.subjects)))

        labels = []
        data = []

        labels += ['n_subjects', 'n_grids']
        data += [n_subjects, n_grids]

        k = list(sorted(metrics))
        labels += k
        data += ['{:.3f}'.format(metrics[k]) for k in k]
        return DataFrame(data, index=labels)

    def plot_n_grid(self, ax):
        N = np.array([len(self.agg.subjects[s]) for s in self.agg.subjects])
        ax.hist(N, density=True)
        ax.set_xlabel('N classifications')
        ax.set_ylabel('Grid Images')
        ax.set_title('Grid Classification Distribution')

    def plot_n_subjects(self, ax):
        N = np.array([len(self.agg.images[s]) for s in self.agg.images])
        ax.hist(N)
        ax.set_xlabel('N classifications')
        ax.set_ylabel('N Subjects')
        ax.set_title('Subject Classification Distribution')

    def load_labels(self):
        database = Database()
        logger.info('Loading labels')

        query = """
            SELECT L.subject_id, L.label
            FROM subject_labels as L
            INNER JOIN image_subjects AS SI
                ON SI.subject_id=L.subject_id
            INNER JOIN images AS I
                ON I.image_id=SI.image_id
            WHERE I.group_id IN (10,11,12,13)
                AND (L.label_name='vegas_cleaned')
        """

        labels = {}
        with database.conn as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                for subject_id, label in tqdm(cursor):
                    if subject_id in self.agg.subjects:
                        labels[subject_id] = label

        return labels

    def load_majority_labels(self, subjects):
        logger.info('Loading majority labels')
        labels = {s: n>0.5 for s, n in tqdm(self.agg.reduce_subjects())}
        labels = {s: labels[s] for s in subjects}
        return labels

    def load_swap_labels(self, subjects):
        logger.info('Loading swap labels')
        labels = {s: n>0.5 for s, n in self.agg.swap.items()}
        labels = {s: labels[s] for s in subjects}
        return labels

    def load_first_label(self, subjects):
        logger.info('Loading first label')
        labels = {s: self.agg.subjects[s][0] for s in subjects}
        return labels



            


        # if self._cleaned:
            # label_name = 'vegas_cleaned'
        # else:
            # label_name = 'vegas'

        reduced_subjects = {s: n for s, n in tqdm(self.agg.reduce_subjects())}

        np_labels = [(labels[s], reduced_subjects[s] > 0.5) for s in labels]
        np_labels = np.array(np_labels).astype(np.int)

        return labels, np_labels

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.load_labels()
        return self._labels

    @property
    def majority(self):
        if self._majority is None:
            self._majority = self.load_majority_labels(self.subjects)
        return self._majority

    @property
    def swap(self):
        if self._swap is None:
            self._swap = self.load_swap_labels(self.subjects)
        return self._swap

    @property
    def first_vote(self):
        if self._first is None:
            self._first = self.load_first_label(self.subjects)
        return self._first

    @property
    def subjects(self):
        if self._subjects is None:
            labeled = set(self.labels)
            subjects = self.agg.subjects
            subjects = set([s for s in subjects if len(subjects[s]) >= 5])

            self._subjects = labeled & subjects
        return self._subjects

    @property
    def np_labels(self):
        if self._np_labels is None:
            subjects = list(self.subjects)
            data = []
            for s in subjects:
                data.append(
                    (self.labels[s], self.majority[s],
                     self.swap[s], self.first_vote[s]))
            self._np_labels = np.array(data)
        return self._np_labels

    def _accuracy(self, index):
        labels = self.np_labels
        accuracy = np.sum(labels[:,0] == labels[:,index]) / labels.shape[0]
        return accuracy

    def _f1(self, index):
        labels = self.np_labels
        f1 = sklearn.metrics.f1_score(labels[:,0], labels[:,index])
        return f1

    def _f1_benchmarks(self):
        labels = self.np_labels
        f1_score = sklearn.metrics.f1_score
        f1_bench = {
            'zeros': f1_score(labels[:,0], np.zeros_like(labels[:,0])),
            'ones': f1_score(labels[:,0], np.ones_like(labels[:,0])),
            'flipped': f1_score(1-labels[:,0], np.ones_like(labels[:,0]))
            }

        return f1_bench

    @property
    def metrics(self):
        if self._metrics is None:
            metrics = {}
            for k, i in [('majority', 1), ('swap', 2), ('first', 3)]:
                metrics[k] = {
                    'accuracy': self._accuracy(i),
                    'f1': self._f1(i),
                }
            metrics['benchmarks'] = self._f1_benchmarks()
            self._metrics = metrics
        return self._metrics

    @property
    def flatten_metrics(self):
        metrics = self.metrics
        for i in metrics:
            for j in metrics[i]:
                yield '{}_{}'.format(j, i), metrics[i][j]

    def plot_summary(self):
        fig = plt.figure(figsize=(12,4))
        fig.subplots_adjust(wspace=0.35)

        ax = fig.add_subplot(121)
        self.plot_n_subjects(ax)
        ax = fig.add_subplot(122)
        self.plot_n_grid(ax)
        plt.show()

        fig = plt.figure(figsize=(12,4))
        fig.subplots_adjust(wspace=0.3)
        ax = fig.add_subplot(121)
        self.plot_n_subjects(ax)
        ax.set_yscale('log')
        ax = fig.add_subplot(122)
        self.plot_n_grid(ax)
        ax.set_yscale('log')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_metrics(ax, 'majority')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_data_breakdown(ax)
        plt.show()

    def plot_metrics(self, ax, key):
        metrics = dict(self.flatten_metrics)
        x = [(k, metrics[k]) for k in sorted(metrics)]
        labels, x = zip(*x)

        plt.barh(labels, x)
        for i, j in enumerate(x):
            plt.text(0.8, i, '{:.3f}'.format(j))
        plt.title('Volunteer Performance')
        plt.grid()

    def plot_data_breakdown(self, ax):
        labels = self.np_labels
        print(labels)
        muons = np.array([np.sum(labels[:,0]==1), np.sum(labels[:,0]==0)])
        print(muons)
        ax.barh(['Muon', 'Non-Muon'], muons)
        pos = np.max(muons)*0.7
        ax.text(pos, 0, '{:.1f}%'.format(muons[0]/labels.shape[0]*100))
        ax.text(pos, 1, '{:.1f}%'.format(muons[1]/labels.shape[0]*100))
        ax.grid()
        ax.set_title('Simulated Event Breakdown')

    @property
    def subject_ids(self):
        return [s for s in self.labels if self.labels[s] == 1]

