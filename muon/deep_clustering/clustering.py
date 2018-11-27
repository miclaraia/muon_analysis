

import os
from time import time
import json
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import f1_score
from sklearn.decomposition import PCA

from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils

import dec_keras as dk
from muon.utils.subjects import Subjects

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
        self.n_classes = kwargs.get('n_classes', 2)
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
    def __init__(self, sid, y, y_pred, subjects, n_clusters):
        self.sid = sid
        self.y = y
        self.y_pred = y_pred
        self._subjects = subjects
        self.n_clusters = n_clusters

        self.cluster_mapping = self._make_cluster_mapping()

    # def subjects(self):
        # return self._subjects.subset(self.sid)

    def cluster_subjects(self, cluster):
        subjects = np.where(self.y_pred == cluster)[0]
        return self._subjects.subset(self.sid[subjects])

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

    def initialize(self, verbose=False):
        config = self.config
        data = self.subjects.get_charge_array(
            order=False, labels=False, rotation=config.rotation)
        x = data[0]
        self.dec.initialize_model(**{
            'optimizer': SGD(lr=config.lr, momentum=config.momentum),
            'ae_weights': config.ae_weights,
            'x': x
        })
        if verbose:
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
        return Prediction(s, y, y_pred, subjects, self.config.n_clusters)

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

    @staticmethod
    def euclidian_distance(vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), \
                                K.epsilon()))

    def cluster_distance(self, x):
        x_encoded = self.dec.encoder.predict(x)
        x_encoded_tiled = np.tile(
            x_encoded[:,:,np.newaxis],
            (1, 1, self.dec.n_clusters)
        )
        cluster_centres = self.get_cluster_centres().T
        cluster_centres_tiled = np.tile(
            cluster_centres[np.newaxis, :, :],
            (x.shape[0], 1, 1)
        )

        def euclidean_distance(vects):
            x, y = vects
            print(x.shape, y.shape)
            N = K.sum(K.square(x - y), axis=1, keepdims=True)
            N = K.sqrt(K.maximum(N, K.epsilon()))
            return N

        euclidean_distances = np.squeeze(K.eval(euclidean_distance(
            (x_encoded_tiled, cluster_centres_tiled)
        )))

        print(euclidean_distances.shape)
        return euclidean_distances

    def get_cluster_assignment(self, x, y):
        cluster_preds = self.dec.predict_clusters(x)
        cluster_distances = self.cluster_distance(x)
        cluster_mapping = list(self.predictions. \
                          cluster_mapping['majority_class'])

        y_assign = []
        for i in range(x.shape[0]):
            l = y[i]
            c = cluster_preds[i]
            if l == cluster_mapping[c]:
                # easy, already the right cluster
                y_assign.append(c)
            else:
                # need to assign to the closest cluster with the right label
                # euclidean distance
                ed = cluster_distances[i][[np.where(cluster_mapping == l)]]
                # assigned cluster
                ac = np.array(cluster_mapping)[np.where(cluster_mapping == l)]
                ac = int(ac[np.argmin(ed)])
                y_assign.append(ac)

        return y_assign

    def get_cluster_centres(self):
        cluster_centers = self.dec.model.get_layer(name='clustering')
        cluster_centers = cluster_centers.get_weights()
        cluster_centers = np.squeeze(np.array(cluster_centers))
        return cluster_centers

    def pca_plot(self):
        x = self.subjects.get_x(False)
        y = self.dec.predict_clusters(x)
        cluster_centers = self.dec.model.get_layer(name='clustering')
        cluster_centers = cluster_centers.get_weights()
        cluster_centers = np.squeeze(np.array(cluster_centers))

        labels = [str(i) for i in range(self.config.n_clusters)]
        return self._pca_plot(x, cluster_centers, y, labels=labels)

    def _pca_plot(self, x, cluster_centres, y=None, labels=[],
                  ulcolour='#747777', ccolour='#4D6CFA'):
        base_network = self.dec.encoder

        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(base_network.predict(x))
        #c_pca = pca.transform(cluster_centres)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        if np.any(y):
            unique_targets = list(np.unique(y))
            cmap = discrete_cmap(len(unique_targets), 'jet')
            norm = matplotlib.colors.BoundaryNorm(
                np.arange(0, max(unique_targets), 1), cmap.N)

            if -1 in unique_targets:
                _x = x_pca[np.where(y == -1), 0]
                _y = x_pca[np.where(y == -1), 1]
                ax.scatter(_x, _y, marker='o', s=20, c=ulcolour, alpha=0.1)
                unique_targets.remove(-1)
            for l in unique_targets:
                _x = x_pca[np.where(y == l), 0]
                _y = x_pca[np.where(y == l), 1]
                _c = l * np.ones(_x.shape)
                ax.scatter(_x, _y, marker='o', s=5, c=_c,
                           cmap=cmap, norm=norm, alpha=0.2, label=labels[l])

        else:
            ax.scatter(x_pca[:,0], x_pca[:,1], marker='o', s=20, \
                color=ulcolour, alpha=0.1)
        plt.axis('off')


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N)
