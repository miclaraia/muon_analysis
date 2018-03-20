import sys
import numpy as np
import matplotlib.pyplot as plt

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


class Mapping:

    def __init__(self, cluster, agg, train, validate):
        subjects = cluster.subjects
        # agg.apply_labels(subjects)
        self.agg = agg
        self.cluster = cluster

        self.train = train
        self.validate = validate
        self.width = subjects.dimensions[1]

        self.classes = [0, 1]
        self.FOM = [cluster.accuracy(validate)]

    def get_xy(self, subset):
        subjects = self.cluster.subjects.subset(subset)
        x, y = subjects.get_charge_array(False, False, True)
        return x, y

    def get_xy_volunteer(self, subset):
        x = []
        y = []
        for s in subset:
            subject = self.cluster.subjects[s]
            for a in self.agg.data['subjects'][s]:
                x.append(subject.charge)
                y.append(a)

        return np.array(x), np.array(y)

    def _cluster_mapping(self):
        pred = self.cluster.predictions
        return list(pred.cluster_mapping['majority_class'])

    def apply_mapping(self):
        n_classes = len(self.classes)
        cluster_mapping = self._cluster_mapping()

        x_train, y_train = self.get_xy(self.train)
        x_val, y_val = self.get_xy(self.validate)

        k_initializer = MapInitializer(cluster_mapping, n_classes)

        a = Input(shape=(self.width,)) # input layer
        q = self.cluster.dec.model(a)

        pred = MappingLayer(
            cluster_mapping, output_dim=n_classes,
            kernel_initializer=k_initializer)(q)

        model = Model(inputs=a, outputs=pred)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        def make_data(x, y):
            return (x, np_utils.to_categorical(y, n_classes))

        x_data = make_data(*self.get_xy_volunteer(self.train))
        val_data = make_data(x_val, y_val)
        model.fit(*x_data, validation_data=val_data, epochs=10, batch_size=256)

        # Generate FOM score
        # y_pred = self.cluster._predict().cluster_mapping['majority_class']
        self.FOM.append(self.cluster.accuracy(self.validate))
        self.FOM.append(self.cluster.accuracy(self.train))

        return self.FOM

    def pca_plot(self):
        lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9',
                    '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']
        x, y = self.get_xy(self.train)
        y = np.array(y, dtype='int')

        cluster_centers = self.cluster.dec.model.get_layer(name='clustering')
        cluster_centers = cluster_centers.get_weights()
        cluster_centers = np.squeeze(np.array(cluster_centers))

        labels = [str(i) for i in range(self.cluster.config.n_clusters)]
        return self._pca_plot(x, cluster_centers, y, labels=labels,
                              lcolours=lcolours)

    def _pca_plot(self, x, cluster_centres, y=None, labels=[],
                 lcolours=[], ulcolour='#747777', ccolour='#4D6CFA'):
        base_network = self.cluster.dec.encoder

        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(base_network.predict(x))
        c_pca = pca.transform(cluster_centres)

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

        if np.any(y):
            unique_targets = list(np.unique(y))
            if -1 in unique_targets:
                _x = x_pca[np.where(y == -1), 0]
                _y = x_pca[np.where(y == -1), 1]
                ax.scatter(_x, _y, marker='o', s=20, color=ulcolour, alpha=0.1)
                unique_targets.remove(-1)
            for l in unique_targets:
                _x = x_pca[np.where(y == 1), 0]
                _y = x_pca[np.where(y == 1), 1]
                ax.scatter(_x, _y, marker='o', s=5, color=lcolours[l],
                           alpha=0.2, label=labels[l])

        else:
            ax.scatter(x_pca[:,0], x_pca[:,1], marker='o', s=20, \
                color=ulcolour, alpha=0.1)
        plt.axis('off')
