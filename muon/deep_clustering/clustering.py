
from muon.utils.subjects import Subjects

import os
from time import time
import json

import numpy as np
from keras.optimizers import SGD
from keras.utils import np_utils
import dec_keras as dk
import matplotlib.pyplot as plt
import pandas as pd


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
    def __init__(self, order, labels, y_pred, subjects, config):
        self.order = np.array(order)
        self.labels = labels
        self.y_pred = y_pred
        self._subjects = subjects
        self.config = config

        self.cluster_mapping = self._make_cluster_mapping()

    def subjects(self):
        subjects = self._subjects
        subjects = [subjects[s] for s in self.order]
        return Subjects(subjects)

    def cluster_subjects(self, cluster):
        subjects = np.where(self.y_pred == cluster)[0]
        subjects = self.order[subjects]
        return self._subjects.subset(subjects)

    def _make_cluster_mapping(self):
        y = self.labels
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

            print(cluster, *data[-1])

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

    def cluster_distance(self):
        order, charge, label = self.subjects.get_charge_array(True)
        X = self._predict(charge)

        clusters = []
        for c in range(self.config.n_clusters):
            distance = self._cluster_distance(X, c)
            cluster = []
            for i, d in distance:
                cluster.append((i, d, order[i], label[i]))

            cluster = np.array(cluster, dtype=[
                ('i', 'i4'),
                ('d', 'f4'),
                ('s', 'i4'),
                ('l', 'i4')])
            clusters.append(cluster)
        return clusters

    def plot_acc(self, c, ax=None, scale='subject'):
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

        ax.plot(x, y)


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
        """
        Initialize the dec model and train
        """
        config = self.config
        _, X = self.subjects.get_charge_array()
        self.dec.initialize_model(**{
            'optimizer': SGD(lr=config.lr, momentum=config.momentum),
            'ae_weights': config.ae_weights,
            'x': X
        })
        print(self.dec.model.summary())

    @property
    def feature_space(self):
        if self._feature_space is None:
            self._feature_space = FeatureSpace(
                self.dec.model,
                self.subjects.labeled_subjects(),
                self.predictions,
                self.config)
        return self._feature_space

    @property
    def predictions(self):
        if self._predictions is None:
            self._predictions = self._predict()
        return self._predictions

    def _predict(self):
        subjects = self.subjects.labeled_subjects()
        order, charges, labels = subjects.get_charge_array(True)
        labels = np.array(labels)

        path = os.path.join(self.config.save_dir, 'DEC_model_final.h5')
        if os.path.isfile(path):
            self.dec.load_weights(path)
            y_pred = self.dec.predict_clusters(charges)
        else:
            y_pred = self._dec_predict(charges, labels)

        return Prediction(order, labels, y_pred,
                          self.subjects, self.config)

    def _dec_predict(self, charges, labels):
        """
        Pass known x and y labels to the dec clustering method
        to train the clustering algorithm
        """

        t0 = time()
        y_pred = self.dec.clustering(charges, y=labels, **{
            'tol': self.config.tol,
            'maxiter': self.config.maxiter,
            'update_interval': self.config.update_interval,
            'save_dir': self.config.save_dir
        })

        print('clustering time: %.2f' % (time() - t0))

        accuracy = dk.cluster_acc(labels, y_pred)
        print(accuracy)
        return y_pred

