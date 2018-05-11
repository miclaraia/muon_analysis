import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
import random

from dec_keras import DEC, cluster_acc

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.initializers import Initializer
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K

from muon.utils.camera import CameraRotate
import muon.deep_clustering.utils as utils


class MapInitializer(Initializer):

    def __init__(self, mapping, n_classes):
        self.mapping = mapping
        self.n_classes = n_classes

    def __call__(self, shape, dtype=None):
        return K.one_hot(self.mapping, self.n_classes)

    def get_config(self):
        return {'mapping': self.mapping, 'n_classes': self.n_classes}

class MappingLayer(Layer):

    def __init__(self, mapping, output_dim, kernel_initializer, **kwargs):
        self.output_dim = output_dim
        # mapping is a list where the index corresponds to a cluster
        # and the value is the label.
        # e.g. say mapping[0] = 5, then a label of 5 has been
        # assigned to cluster 0

        # get the number of classes
        self.n_classes = np.unique(mapping).shape[0]
        self.mapping = K.variable(mapping, dtype='int32')
        self.kernel_initializer = kernel_initializer
        super(MappingLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel', 
            shape=(input_shape[1], self.output_dim),
            initializer=self.kernel_initializer,
            trainable=False)

        # Be sure to call this somewhere!
        super(MappingLayer, self).build(input_shape)  

    def call(self, x):
        return K.softmax(K.dot(x, self.kernel))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# class MappingSubset:
    # def __init__(self, subset, subjects, labels, truth, threshold):
        # self.subset = subset
        # subjects = subjects.subset(subset)
        # self.x, self.y = subjects.get_xy(labels, False)
        # _, self.y_truth = subjects.get_xy(truth, False)

        # self.threshold = threshold
        # self.score = None
        # self.y_pred = None
        # self.pred_cluster = None
        # self.pred_class = None

    # def predict(self, model, cluster):
        # truth = self.y_truth
        # pred = model.predict(self.x)[:,1]
        # self.y_pred = pred

        # index = list(range(len(pred)))
        # index = sorted(index, key=lambda i: pred[i])
        # truth = [truth[i] for i in index]
        # for i, _ in enumerate(truth):
            # purity = sum(truth[i:])/len(truth[i:])
            # if purity >= self.threshold:
                # self.score = {'precision': purity,
                              # 'recall': sum(truth[i:])/sum(truth)}
                # break
        
        # pred = cluster.predict(self.subset)
        # self.pred_cluster = pred.y_pred
        # self.pred_class = pred.predict_class
        # return self.score

    # def get_xy(self):
        # return self.x, self.y

    # def accuracy(self):
        # print(self.y, self.y_pred)
        # return f1_score(self.y, self.pred_class)


class Mapping:

    def __init__(self, cluster, labels, truth, train, validate,
                 threshold):
        subjects = cluster.subjects
        self.truth = truth
        self.cluster = cluster

        self.train = utils.SubjectSubset(
            train, subjects, labels, truth, threshold)
        self.validate = utils.SubjectSubset(
            validate, subjects, labels, truth, threshold)

        self.width = subjects.dimensions[1]

        self.classes = [0, 1]
        self.FOM = []
        self.model = None

        self.scores = []

    def score(self):
        self.validate.predict(self.model, self.cluster)
        self.train.predict(self.model, self.cluster)
        self.scores.append({'train': self.train.score,
                            'validate': self.validate.score})

    def cluster_mapping(self):
        pred = self.cluster.predict()
        cm = pred.cluster_mapping
        return cm[cm['majority_class_fraction'] >= 0]

    def init_model(self, loss, optimizer):
        n_classes = len(self.classes)

        cluster_mapping = list(self.cluster.predictions. \
                          cluster_mapping['majority_class'])

        k_initializer = MapInitializer(cluster_mapping, n_classes)
        a = Input(shape=(self.width,)) # input layer
        q = self.cluster.dec.model(a)

        pred = MappingLayer(
            cluster_mapping, output_dim=n_classes,
            kernel_initializer=k_initializer)(q)

        model = Model(inputs=a, outputs=pred)
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

        self.score()

        return model

    def apply_mapping(self, fit_callback, xy_callback=None):
        self.fit(fit_callback, xy_callback)

    def fit(self, fit_callback, xy_callback=None):
        if xy_callback:
            train = xy_callback(self.train, self.cluster.subjects)
        else:
            train = self.train.get_xy()
        val = self.validate.get_xy()

        def make_data(xy):
            x, y = xy
            return (x, np_utils.to_categorical(y, 2))

        val = make_data(val)
        train = make_data(train)

        fit_callback(self.model, train, val)
        self.score()

    def pca_plot(self):
        self.cluster.pca_plot()


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return LinearSegmentedColormap.from_list(cmap_name, color_list, N)
