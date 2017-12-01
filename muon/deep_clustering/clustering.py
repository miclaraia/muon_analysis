
from muon.utils.subjects import Subjects

import os
from time import time

import numpy as np
from keras.optimizers import SGD
from keras.utils import np_utils
import dec_keras as dk


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
    def __init__(self, save_dir, ae_weights, **kwargs):
        self.n_clusters = kwargs.get('n_clusters', 10)
        self.batch_size = kwargs.get('batch_size', 256)
        self.nodes = kwargs.get('nodes', [500, 500, 2000, 10])
        self.lr = kwargs.get('lr', .01)
        self.momentum = kwargs.get('momentum', .9)
        self.tol = kwargs.get('tol', .001)
        self.maxiter = kwargs.get('maxiter', 2e4)
        self.update_interval = kwargs.get('update_interval', 140)
        self.save_dir = save_dir
        self.ae_weights = ae_weights

class Prediction:
    def __init__(self, order, labels, y_pred, config):
        self.order = order
        self.labels = labels
        self.y_pred = y_pred
        self.config = config

        self.cluster_mapping = self._make_cluster_mapping()

    def subjects(self, subjects):
        subjects = [subjects[s] for s in self.order]
        return Subjects(subjects)

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

        data = zip(*data)
        print(data)

        # cluster_mapping, n_assigned_list, majority_class_fraction
        return data


class Cluster:

    def __init__(self, dec, subjects, config):
        self.dec = dec
        self.subjects = subjects
        self.config = config
        self._predictions = None

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

        return Prediction(order, labels, y_pred, self.config)

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







