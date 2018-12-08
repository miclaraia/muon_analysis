import os
import numpy as np

from keras.optimizers import SGD
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import homogeneity_score

import dec_keras.DEC
import muon.deep_clustering.clustering
from muon.dissolving.utils import get_cluster_to_label_mapping_safe, \
        calc_f1_score, one_percent_fpr
from muon.dissolving.utils import Metrics
import muon.dissolving.utils


class Config(muon.dissolving.utils.Config):
    pass


class DECv2(dec_keras.DEC):

    def __init__(self, config, input_shape):
        super().__init__(
            n_clusters=config.n_clusters,
            dims=[input_shape[1]] + config.nodes,
            batch_size=config.batch_size)

        print(self.dims)

        self.n_classes = config.n_classes
        self.n_clusters = config.n_clusters
        self.config = config

    def init(self, x_train, verbose=True):
        ae_weights, dec_weights = self.config.save_weights

        self.initialize_model(
            optimizer=self.config.get_optimizer(),
            ae_weights=ae_weights,
            x=x_train)

        if os.path.isfile(dec_weights):
            self.model.load_weights(dec_weights, by_name=True)
        if verbose:
            print(self.model.summary())

    @classmethod
    def load(cls, save_dir, x_train, verbose=True):
        config = Config.load(os.path.join(save_dir, 'config.json'))
        input_shape = x_train.shape

        self = cls(config, input_shape)
        self.init(x_train, verbose)

        return self

    def _calculate_metrics(self, x, y, y_pred, c_map):
        cluster_pred = self.model.predict(x, verbose=0)[1].argmax(1)
        f1 = f1_score(y[:,1], np.argmax(y_pred, axis=1))
        f1c = calc_f1_score(y[:,1], cluster_pred, c_map)
        h = homogeneity_score(y[:,1], cluster_pred)
        nmi = metrics.normalized_mutual_info_score(y[:,1], cluster_pred)

        return (f1, f1c, h, nmi)

    def clustering(
            self,
            train_data,
            train_dev_data,
            validation_data):
        xy = np.concatenate((train_data, train_dev_data))
        x, y = xy[:, 0], xy[:, 1]

        super().clustering(
            x, y=y,
            tol=self.config.tol,
            maxiter=self.config.maxiter,
            update_interval=self.config.update_interval,
            save_dir=self.config.save_dir)
