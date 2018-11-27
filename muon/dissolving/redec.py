import numpy as np
import os

from sklearn import metrics
from sklearn.metrics import homogeneity_score

# from keras.models import Model
# from keras.layers import Input, Dense
# from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
# from keras import backend as K
# from keras import regularizers
# from keras.optimizers import SGD
# from keras.models import load_model

from dec_keras.DEC import DEC, ClusteringLayer, cluster_acc
from muon.dissolving.utils import get_cluster_to_label_mapping_safe, \
        calc_f1_score, one_percent_fpr
from muon.dissolving.utils import Metrics
import muon.dissolving.utils


class Config(muon.dissolving.utils.Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_interval = kwargs.get('save_interval') or 5
        self.update_interval = kwargs.get('update_interval') or 140


class ReDEC(DEC):
    def __init__(self, metrics, config, input_shape):
        super().__init__(
            n_clusters=config.n_clusters,
            dims=[input_shape[1]] + config.nodes,
            batch_size=config.batch_size)

        print(self.dims)

        self.metrics = metrics
        self.n_classes = config.n_classes
        self.n_clusters = config.n_clusters
        self.config = config

    def init(self, x_train):
        ae_weights, dec_weights = self.config.save_weights
        print(x_train.shape)
        self.initialize_model(
            optimizer=self.config.get_optimizer(),
            ae_weights=ae_weights,
            x=x_train)

        self.model.load_weights(dec_weights, by_name=True)
        print(self.model.summary())

    @classmethod
    def load(cls, save_dir, x_train):
        config = Config.load(os.path.join(save_dir, 'config.json'))
        input_shape = x_train.shape

        self = cls(None, config, input_shape)
        self.init(x_train)
        print(self.model.summary())

        return self

    def load_multitask_weights(self, mdec):
        for i in range(1, len(mdec.model.layers[1].layers)):
            self.model.layers[i].set_weights(
                mdec.model.layers[1].layers[i].get_weights())
        self.model.layers[-1].set_weights(mdec.model.layers[2].get_weights())

    def _calculate_metrics(self, x, y, c_map):
        cluster_pred = self.model.predict(x, verbose=0).argmax(1)
        f1c = calc_f1_score(y, cluster_pred, c_map)
        h = homogeneity_score(y, cluster_pred)
        nmi = metrics.normalized_mutual_info_score(y, cluster_pred)

        return np.nan, f1c, h, nmi

    def get_cluster_map(self, x, y, toprint=False):
        train_q = self.model.predict(x, verbose=0)
        # train_p = self.target_distribution(train_q)
        c_map = get_cluster_to_label_mapping_safe(
            y, train_q.argmax(1), self.n_classes, self.n_clusters,
            toprint=toprint)

        return c_map

    def clustering(self,
                   train_data,
                   train_dev_data,
                   test_data,
                   valid_data):

        tol = self.config.tol
        update_interval = self.config.update_interval
        save_interval = self.config.save_interval
        epochs = self.config.maxiter

        save_dir = self.config.save_dir

        x = np.concatenate((train_data[0], train_dev_data[0]))
        y = np.concatenate((train_data[1], train_dev_data[1]))

        x_valid, y_valid = valid_data

        y_pred_last = None
        # iterations per epoch
        batches = np.ceil(x.shape[0] / self.batch_size)

        print('Update interval', update_interval)
        save_interval = batches * 5
        print('Save interval', save_interval)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self.metrics is None:
            self.metrics = Metrics()

        loss = 0
        index = 0
        for ite in range(int(epochs * batches)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                # update the auxiliary target distribution p
                p = self.target_distribution(q)

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if ite > 0:
                    delta_label = np.sum(y_pred != y_pred_last).\
                            astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                if valid_data is not None:
                    train_q = self.model.predict(x, verbose=0)
                    train_p = self.target_distribution(train_q)
                    c_map, _, _ = get_cluster_to_label_mapping_safe(
                        y, train_q.argmax(1), self.n_classes, self.n_clusters,
                        toprint=False)

                    metrics_train = \
                        self._calculate_metrics(x, y, c_map)

                    metrics_valid = \
                        self._calculate_metrics(x_valid, y_valid, c_map)

                    valid_q = self.model.predict(x_valid, verbose=0)
                    valid_p = self.target_distribution(valid_q)
                    val_loss = self.model.test_on_batch(x_valid, valid_p)
                    metrics_valid = \
                        self._calculate_metrics(x_valid, y_valid, c_map)

                    print(self.metrics.add(
                        ite/batches,
                        metrics_train, metrics_valid, loss, val_loss))

                # check stop criterion
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # train on batch
            if (index + 1) * self.batch_size > x.shape[0]:
                loss = self.model.train_on_batch(
                        x=x[index * self.batch_size::],
                        y=p[index * self.batch_size::])
                index = 0
            else:
                loss = self.model.train_on_batch(
                        x=x[index * self.batch_size:(index + 1) * self.batch_size],
                        y=p[index * self.batch_size:(index + 1) * self.batch_size])
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                fname = os.path.join(save_dir, 'DEC_model_{}.h5'.format(ite))
                print('saving model to: {}'.format(fname))
                self.model.save_weights(fname)

            if ite % update_interval == 0:
                self.metrics.save(os.path.join(
                    save_dir, 'metrics_intermediate.pkl'))

        # save the trained model
        print('saving model to:', save_dir + '/DEC_model_final.h5')
        self.model.save_weights(save_dir + '/DEC_model_final.h5')

        return y_pred, self.metrics
