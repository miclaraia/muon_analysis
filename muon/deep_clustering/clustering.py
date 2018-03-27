
from muon.utils.subjects import Subjects

import os
from time import time
import json

import numpy as np
from sklearn.metrics import f1_score
from keras.optimizers import SGD
from keras.utils import np_utils
import dec_keras as dk
import matplotlib.pyplot as plt
import pandas as pd

import logging
logger = logging.getLogger(__name__)

# this is chosen based on prior knowledge of classes in the data set.
# n_clusters = 10
# batch_size = 256
# # learning rate
# lr         = 0.01
# momentum   = 0.9
# # tolerance - if clustering stops if less than this fraction
# # of the data changes cluster on an interation
# tol        = 0.001

# maxiter         = 2e4
# update_interval = 140
# save_dir        = '../DEC-keras/results/'
# ae_weights = '../DEC-keras/results/mnist/ae_weights.h5'

class Config:
    def __init__(self, save_dir, **kwargs):
        self.n_clusters = kwargs.get('n_clusters', 10)
        self.batch_size = kwargs.get('batch_size', 256)
        self.nodes = kwargs.get('nodes', [500, 500, 2000, 10])
        self.lr = kwargs.get('lr', .01)
        self.momentum = kwargs.get('momentum', .9)
        self.tol = kwargs.get('tol', .001)
        self.maxiter = kwargs.get('maxiter', 2e4)
        self.update_interval = kwargs.get('update_interval', 140)
        self.rotation = kwargs.get('rotation', False)

        self.save_dir = os.path.abspath(save_dir)

        subjects = os.path.join(save_dir, 'subjects.pkl')
        subjects = kwargs.get('subjects', subjects)
        self.subjects = os.path.abspath(subjects)

        ae_weights = kwargs.get('ae_weights', None)
        if ae_weights is None:
            ae_weights = os.path.join(save_dir, 'ae_weights.h5')
        self.ae_weights = os.path.abspath(ae_weights)

    def dump(self):
        fname = os.path.join(self.save_dir, 'config.json')
        json.dump(self.__dict__, open(fname, 'w'))

    @classmethod
    def load(cls, fname):
        data = json.load(open(fname, 'r'))
        config = cls('')
        config.__dict__.update(data)
        return config

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)


class Prediction:
    """
    Stores and analyzes machine predictions given the real subject labels
    """
    def __init__(self, order, y, y_pred, subjects, config):
        self.order = np.array(order)
        self.y = y
        self.y_pred = y_pred
        self._subjects = subjects
        self.config = config

        # self.cluster_mapping = self._make_cluster_mapping()
        self.cluster_mapping = self._make_cluster_mapping()

    def subjects(self):
        subjects = self._subjects
        subjects = [subjects[s] for s in self.order]
        return Subjects(subjects)

    def cluster_subjects(self, cluster):
        subjects = np.where(self.y_pred == cluster)[0]
        subjects = self.order[subjects]
        return self._subjects.subset(subjects)

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
            y = [-1 for i in self.order]

        y_pred = self.y_pred
        n_classes = len(np.unique(y))
        n_clusters = self.config.n_clusters

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


class FeatureSpace:
    """
    Analyze how the clusters are structured. Looking at how far away
    each subject is from the cluster center
    """
    def __init__(self, model, subjects, pred, config):
        self.model = model
        self.subjects = subjects
        self.config = config
        self.prediction = pred

        self.clusters = self.cluster_distance()

    def _predict(self, charge):
        return self.model.predict(charge)

    def _cluster_distance(self, X, cluster):
        nearest = np.where(X.argmax(axis=1) == cluster)[0]
        data = []
        for i in nearest:
            data.append((i, X[i][cluster]))

        data = sorted(data, key=lambda i: i[1])
        return data

    def closest_subjects(self, cluster, size):
        cluster_size = self.prediction.cluster_mapping['n_assigned'][cluster]
        if size == 'all':
            size = cluster_size
        else:
            size = min(cluster_size, size)

        subjects = self.clusters[cluster]['s'][:size]
        return self.subjects.subset(subjects)

    def cluster_subjects(self, cluster):
        subjects = self.clusters[cluster]['s']
        return self.subjects.subset(subjects)

    def cluster_distance(self):
        order, charge = self.subjects.get_charge_array()
        X = self._predict(charge)

        clusters = []
        for c in range(self.config.n_clusters):
            distance = self._cluster_distance(X, c)
            cluster = []
            for i, d in distance:
                s = self.subjects[order[i]]
                cluster.append((i, d, s.id))

            clusters.append(pd.DataFrame(cluster, columns=['i', 'd', 's']))
        return clusters

    def plot_acc(self, cluster, ax=None, scale='subject', **kwargs):
        c = cluster
        cluster = self.clusters[c]
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        if scale == 'subject':
            x = range(cluster.shape[0])
        elif scale == 'distance':
            x = cluster['d']
        y = []

        majority = self.prediction.cluster_mapping['majority_class'][c]
        n = 0
        for i, l in enumerate(cluster['l']):
            if l == majority:
                n += 1
            y.append(n/(i+1))

        ax.plot(x, y, **kwargs)


class Cluster:

    def __init__(self, dec, subjects, config):
        self.dec = dec
        self.subjects = subjects
        self.config = config
        self._predictions = None
        self._feature_space = None

    @classmethod
    def create(cls, subjects, config):
        dec = dk.DEC(
            dims=[subjects.dimensions[1]] + config.nodes,
            n_clusters=config.n_clusters,
            batch_size=config.batch_size
        )

        return cls(dec, subjects, config)

    def train(self):
        raise DeprecationWarning('Deprecated, use initialize()')

    def initialize(self):
        config = self.config
        data = self.subjects.get_charge_array(
            order=False, labels=False, rotation=config.rotation)
        x = data[0]
        self.dec.initialize_model(**{
            'optimizer': SGD(lr=config.lr, momentum=config.momentum),
            'ae_weights': config.ae_weights,
            'x': x
        })
        print(self.dec.model.summary())

        # Try to load clustering weights
        path = os.path.join(self.config.save_dir, 'DEC_model_final.h5')
        if os.path.isfile(path):
            self.dec.load_weights(path)

    def train_clusters(self):
        """
        Train the clustering layer
        """
        data = self.subjects.get_charge_array(
            labels=True, order=False, rotation=self.config.rotation)
        x, y = data[:2]

        t0 = time()
        y_pred = self.dec.clustering(x, y=y, **{
            'tol': self.config.tol,
            'maxiter': self.config.maxiter,
            'update_interval': self.config.update_interval,
            'save_dir': self.config.save_dir
        })

        print('clustering time: %.2f' % (time() - t0))

        # TODO doesn't work without labels
        if y is not None:
            accuracy = dk.cluster_acc(y, y_pred)
            print(accuracy)
        else:
            print("No labels to predict clustering accuracy")
        return y_pred

    def predict(self, subset=None):
        """
        Predict cluster for a set of subjects
        """
        subjects = self.subjects
        if subset:
            subjects = subjects.subset(subset)

        data = subjects.get_charge_array(
            labels=True, order=True, rotation=False)
        s, x, y = data[:3]

        if -1 in y:
            logger.warning('found -1 in labels, not using labels')
            y = None
        y_pred = self.dec.predict_clusters(x)
        return Prediction(s, y, y_pred, subjects, self.config)

    @property
    def feature_space(self):
        if self._feature_space is None:
            self._feature_space = FeatureSpace(
                self.dec.model,
                self.subjects,
                self.predictions,
                self.config)
        return self._feature_space

    @property
    def predictions(self):
        if self._predictions is None:
            self._predictions = self.predict()
        return self._predictions

    def accuracy(self, subset=None):
        subjects = self.subjects
        if subset:
            subjects = subjects.subset(subset)
        x, y = subjects.get_charge_array(
            order=False, labels=True, rotation=False)
        y_pred = self.predict(subset).predict_class

        return f1_score(y, y_pred)
