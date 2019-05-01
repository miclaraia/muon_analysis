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

        self.agg = agg
        self._labels = None
        self._metrics = None
        self._cleaned = cleaned

    def summary(self):
        agg = self.agg
        n_subjects = np.sum([len(agg.subjects[s]) for s in agg.subjects])
        n_grids = np.sum([len(agg.images[s]) for s in agg.images])
        metrics = self.metrics
        print(metrics)

        labels = []
        data = []

        labels += ['n_subjects', 'n_grids']
        data += [n_subjects, n_grids]

        metric_k = list(sorted(metrics.keys()))
        labels += metric_k
        data += ['{:.3f}'.format(metrics[k]) for k in metric_k]
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
            SELECT subject_labels.subject_id, subject_labels.label
            FROM subject_labels
            INNER JOIN image_subjects
                ON image_subjects.subject_id=subject_labels.subject_id
            WHERE image_subjects.group_id=13
                AND subject_labels.label_name=?
        """

        labels = {}

        if self._cleaned:
            label_name = 'vegas_cleaned'
        else:
            label_name = 'vegas'

        reduced_subjects = {s: n for s, n in tqdm(self.agg.reduce_subjects())}
        with database.conn as conn:
            cursor = conn.execute(query, (label_name,))
            for subject_id, label in tqdm(cursor):
                if subject_id in reduced_subjects:
                    labels[subject_id] = label

        np_labels = [(labels[s], reduced_subjects[s] > 0.5) for s in labels]
        np_labels = np.array(np_labels).astype(np.int)

        return labels, np_labels

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.load_labels()
        return self._labels[0]

    @property
    def np_labels(self):
        if self._labels is None:
            self._labels = self.load_labels()
        return self._labels[1]

    def _accuracy(self):
        labels = self.np_labels
        accuracy = np.sum(labels[:,0] == labels[:,1]) / labels.shape[0]
        return accuracy

    def _f1(self):
        labels = self.np_labels
        f1 = sklearn.metrics.f1_score(labels[:,0], labels[:,1])
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
            self._metrics = {
                'accuracy': self._accuracy(),
                'f1': self._f1(),
                **{'f1_{}'.format(k): v for k, v in
                    self._f1_benchmarks().items()}
            }
        return self._metrics

    def plot_metrics(self, ax):
        metrics = self.metrics
        x = [metrics[k] for k in ['accuracy', 'f1', 'f1_ones']]
        labels = ['Accuracy', 'F1 Score', 'F1 Benchmark (all ones)']

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

