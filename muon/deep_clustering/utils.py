


import os
import numpy as np
import pandas as pd
import csv
from collections import OrderedDict

from sklearn.metrics import f1_score
from keras.utils import np_utils

import logging
logger = logging.getLogger(__name__)


class SubjectSubset:
    def __init__(self, subset, subjects, labels, truth, threshold):
        self.subset = subset
        subjects = subjects.subset(subset)
        self.x, self.y = subjects.get_xy(labels, False)
        _, self.y_truth = subjects.get_xy(truth, False)

        self.labels = labels
        self.truth = truth

        self.threshold = threshold
        self.score = None
        self.y_pred = None
        self.pred_cluster = None
        self.pred_class = None

    def predict(self, model, cluster):
        pred = Prediction.generate(cluster, self.subset, self.labels)
        self.pred_cluster = pred.y_pred
        self.pred_class = pred.predict_class

        truth = self.y_truth
        pred = model.predict(self.x)[:, 1]
        self.y_pred = pred

        index = list(range(len(pred)))
        index = sorted(index, key=lambda i: pred[i])
        truth = [truth[i] for i in index]
        score = {'precision': 0, 'recall': 0}
        for i, _ in enumerate(truth):
            purity = sum(truth[i:])/len(truth[i:])
            if purity >= self.threshold:
                score = {'precision': purity,
                         'recall': sum(truth[i:])/sum(truth)}
                break
        score['f1_score'] = self.f1_score()
        self.score = score

        return self.score

    def get_xy(self):
        return self.x, self.y

    def f1_score(self):
        return f1_score(self.y, self.pred_class)


class Prediction:
    """
    Stores and analyzes machine predictions given the real subject labels
    """
    def __init__(self, subject_ids, y, y_pred, subjects, n_clusters):
        self.sid = subject_ids
        self.y = y
        self.y_pred = y_pred
        self.n_clusters = n_clusters

        self.cluster_mapping = self._make_cluster_mapping()

    @classmethod
    def generate(cls, cluster, subset, labels):
        subjects = cluster.subjects
        if subset:
            subjects = subjects.subset(subset)
        x, y = subjects.get_xy(labels, False)
        s = subjects.keys()

        if -1 in y:
            logger.warning('found -1 in labels, not using labels')
            y = None
        y_pred = cluster.dec.predict_clusters(x)
        return cls(s, y, y_pred, subjects, cluster.config.n_clusters)
        

    def cluster_subjects(self, cluster, subjects):
        """
        Return subset of subjects in a cluster
        """
        subset = np.where(self.y_pred == cluster)[0]
        return subjects.subset(self.sid[subset])

    @property
    def predict_class(self):
        pred = np.zeros(len(self.y_pred))
        mapping = self.cluster_mapping['majority_class']
        for i, c in enumerate(self.y_pred):
            pred[i] = mapping[c]

        return pred

    def _make_cluster_mapping(self):
        y = self.y
        if y is None:
            y = [-1 for i in self.sid]

        y_pred = self.y_pred
        n_classes = len(np.unique(y))
        n_clusters = self.n_clusters

        one_hot_encoded = np_utils.to_categorical(y, n_classes)

        # Data will take shape
        # [(number of subjects in cluster,
        #   cluster mapping (majority cluster class),
        #   fraction of majority in cluster)]
        data = []

        for cluster in range(n_clusters):
            # indices of subjects assigned to this cluster
            cluster_indices = np.where(y_pred == cluster)[0]
            n_assigned_examples = cluster_indices.shape[0]

            cluster_labels = one_hot_encoded[cluster_indices]
            cluster_label_fractions = np.mean(cluster_labels, axis=0)

            # Most frequent true label in this cluster
            majority_cluster_class = np.argmax(cluster_label_fractions)

            data.append((cluster_indices.shape[0],
                         majority_cluster_class,
                         # Fraction of the majority class in this cluster
                         cluster_label_fractions[majority_cluster_class]))

            # print(cluster, *data[-1])

        # cluster_mapping, n_assigned_list, majority_class_fraction
        keys = ['n_assigned',
                'majority_class',
                'majority_class_fraction']
        # data = zip(*data)
        # data = {keys[i]:v for i, v in enumerate(data)}
        return pd.DataFrame(data, columns=keys)

    def __str__(self):
        return str(self.cluster_mapping)


def checkpoint_name(path):
    n = 0
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            if 'checkpoint' in f:
                a = f.find('_')+1
                b = f.find('.h5')
                n = int(f[a:b])
    f = os.path.join(path, 'checkpoint_%d.h5' % (n+1))
    return f


def load_set(fname):
    with open(fname, 'r') as file:
        reader = csv.DictReader(file)
        return [int(item['subject']) for item in reader]


def score(y_prob, y_true, threshold):
    index = list(range(len(y_prob)))
    index = sorted(index, key=lambda i: y_prob[i])
    truth = [y_true[i] for i in index]
    score = {'precision': 0, 'recall': 0}
    for i, _ in enumerate(truth):
        purity = sum(truth[i:])/len(truth[i:])
        if purity >= threshold:
            score = {'precision': purity,
                     'recall': sum(truth[i:])/sum(truth)}
            break

    score['f1_score'] = f1_score(y_true, (y_prob>.5).astype(int))
    return score


def pd_scores(scores):
    d = []
    for s in scores:
        d.append((
            s['train']['precision'],
            s['train']['recall'],
            s['train']['f1_score'],
            s['validate']['precision'],
            s['validate']['recall'],
            s['validate']['f1_score'],
        ))
        keys = ['t_precision', 't_recall', 't_f1',
                'v_precision', 'v_recall', 'v_f1']
    return pd.DataFrame(d, columns=keys)

