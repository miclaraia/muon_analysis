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


class MappingSubset:
    def __init__(self, subset):
        self.subset = subset
        self.y_pred = None
        self.x = None
        self.y = None
        self.pred_cluster = None
        self.pred_class = None

    def predict(self, cluster):
        x, y = cluster.subjects.subset(self.subset).get_charge_array(
            order=False, labels=True, rotation=False)
        self.x = x
        self.y = y
        pred = cluster.predict(self.subset)
        self.pred_cluster = pred.y_pred
        self.pred_class = pred.predict_class

    def get_xy(self):
        return self.x, self.y

    def accuracy(self):
        print(self.y, self.y_pred)
        return f1_score(self.y, self.pred_class)


class Mapping:

    def __init__(self, cluster, agg, train, validate,
                 mode='subsample', size=1000, batched=False, shuffle=False,
                 weights=(0, 1)):
        subjects = cluster.subjects
        # agg.apply_labels(subjects)
        self.agg = agg
        self.cluster = cluster

        self.mode = {'mode': mode, 'size': size, 'batched': batched,
                     'shuffle': shuffle, 'weights': weights}

        self.train = MappingSubset(train)
        self.validate = MappingSubset(validate)

        self.train.predict(cluster)
        self.validate.predict(cluster)

        self.width = subjects.dimensions[1]

        self.classes = [0, 1]
        self.FOM = []

        self.score(self.validate)
        self.score(self.train)

    def score(self, subset):
        self.FOM.append(subset.accuracy())

    def get_xy_volunteer(self, subset):
        x = []
        y = []
        mode = self.mode['mode']
        size = self.mode['size']
        def get_xy(subset):
            x = []
            y = []
            for s in subset:
                subject = self.cluster.subjects[s]
                for a in self.agg.data['subjects'][s]:
                    x.append(s)
                    y.append(a)
            print(len(x))
            return np.array(list(zip(x, y)))

        def out(data):
            x, y = zip(*data)
            x = [self.cluster.subjects[s].charge for s in x]
            return np.array(x), np.array(y)

        if mode == 'aggregate':
            reduce = self.agg.reduce()[1]
            data = np.zeros((len(subset), 2))
            for i, s in enumerate(subset):
                subject = self.cluster.subjects[s]
                data[i] = (s, int(reduce[s]>.5))

        elif mode == 'weightedagg':
            reduce = self.agg.reduce()[1]
            x = []
            y = []
            a, b = self.mode['weights']
            for s in subset:
                subject = self.cluster.subjects[s]
                if reduce[s] == a or reduce[s] == b:
                    x.append(s)
                    y.append(int(reduce[s]>.5))
            data = np.array(list(zip(x, y)))

        elif mode == 'subsample':
            data = get_xy(subset)
            sample = random.sample(range(len(data)), size)
            data = data[sample, :]

        elif mode == 'all':
            data = get_xy(subset)

        if self.mode['shuffle']:
            np.random.shuffle(data)

        if self.mode['batched']:
            print('batching')
            while len(data) > 0:
                b = min(len(data), size)
                print(len(data), b)
                yield out(data[:b,:])
                data = data[b:,:]
            return
        else:
            print('not batching')
            yield out(data)

    def _cluster_mapping(self):
        pred = self.cluster.predictions
        return list(pred.cluster_mapping['majority_class'])

    def _create_mapping_model(self):
        n_classes = len(self.classes)
        cluster_mapping = self._cluster_mapping()

        k_initializer = MapInitializer(cluster_mapping, n_classes)
        a = Input(shape=(self.width,)) # input layer
        q = self.cluster.dec.model(a)

        pred = MappingLayer(
            cluster_mapping, output_dim=n_classes,
            kernel_initializer=k_initializer)(q)

        model = Model(inputs=a, outputs=pred)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model


    def apply_mapping(self, epochs=10):
        n_classes = len(self.classes)
        x_train, y_train = self.train.get_xy()
        x_val, y_val = self.validate.get_xy()

        model = self._create_mapping_model()

        def make_data(x, y):
            return (x, np_utils.to_categorical(y, n_classes))

        val_data = make_data(x_val, y_val)

        tracking = []
        for batch in self.get_xy_volunteer(self.train.subset):
            x_batch = make_data(*batch)
            print('batch', len(x_batch))
            model.fit(*x_batch, validation_data=val_data,
                      epochs=epochs, batch_size=256)

            self.validate.predict(self.cluster)
            tracking.append(self.validate.accuracy())
            print('current validation: ', tracking[-1])
        self.tracking = tracking
        print(tracking)

        # Generate FOM score
        self.validate.predict(self.cluster)
        self.score(self.validate)

        self.train.predict(self.cluster)
        self.score(self.train)

        return self.FOM

    def pca_plot(self):
        x = self.train.x
        y = self.train.pred_cluster
        y = np.array(y, dtype='int')

        cluster_centers = self.cluster.dec.model.get_layer(name='clustering')
        cluster_centers = cluster_centers.get_weights()
        cluster_centers = np.squeeze(np.array(cluster_centers))

        labels = [str(i) for i in range(self.cluster.config.n_clusters)]
        return self._pca_plot(x, cluster_centers, y, labels=labels)

    def _pca_plot(self, x, cluster_centres, y=None, labels=[],
                 ulcolour='#747777', ccolour='#4D6CFA'):
        base_network = self.cluster.dec.encoder

        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(base_network.predict(x))
        c_pca = pca.transform(cluster_centres)

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

        if np.any(y):
            unique_targets = list(np.unique(y))
            cmap = discrete_cmap(len(unique_targets), 'jet')
            norm = matplotlib.colors.BoundaryNorm(
                np.arange(0, max(unique_targets),1), cmap.N)

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
