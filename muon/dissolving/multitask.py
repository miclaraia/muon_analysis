import numpy as np
import os
import logging
from tqdm import tqdm
import pickle

from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import homogeneity_score

from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras import regularizers
# from keras.models import load_model

from dec_keras.DEC import DEC, ClusteringLayer, cluster_acc
from muon.dissolving.utils import get_cluster_to_label_mapping_safe, \
        calc_f1_score, one_percent_fpr

lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9',
            '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']

logger = logging.getLogger(__name__)

#  DEC constants from DEC paper
# batch_size = 256
# lr         = 0.01
# momentum   = 0.9
# tol        = 0.001
# maxiter    = 100
# update_interval = 140 #perhaps this should be 1 for multitask learning
# # update_interval = 10 #perhaps this should be 1 for multitask learning
# n_clusters = 10 # number of clusters to use
# n_classes  = 2  # number of classes

class MyLossWeightCallback(Callback):
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    # customize your behavior
    def on_epoch_begin(self, epoch, logs={}):
        self.alpha = self.alpha
        self.beta = self.beta
        self.gamma = self.gamma


class Metrics:
    _metric_names = ['f1', 'f1c', 'h', 'nmi']

    def __init__(self):
        self.metrics = []

    @classmethod
    def _print(cls, metric):

        s = '-------------------------------------------------\n' \
            '%4d  F1=%.4f  F1c=%.4f  h=%.4f  nmi=%.4f\n' \
            '     vF1=%.4f vF1c=%.4f vh=%.4f vnmi=%.4f\n' \
            '      loss=%s\n' \
            '     vloss=%s\n' \
            '-------------------------------------------------\n'

        m = (metric['iteration'],
             *(metric['train'][f] for f in cls._metric_names),
             *(metric['valid'][f] for f in cls._metric_names),
             metric['loss'], metric['vloss'])
        return s % m

    def print_ite(self, iteration):
        for item in self.metrics:
            if item['iteration'] == iteration:
                return self._print(item)

    def add(self, iteration, train, valid, loss, vloss):
        metrics = {
            'iteration': iteration,
            'train': {f: train[i] for i, f in enumerate(self._metric_names)},
            'valid': {f: valid[i] for i, f in enumerate(self._metric_names)},
            'loss': loss,
            'vloss': vloss
        }
        self.metrics.append(metrics)
        return self._print(metrics)

    def dump(self):
        output = {k: [] for k in [
            'iteration',
            'train_f1',
            'train_f1c',
            'train_h',
            'train_nmi',
            'valid_f1',
            'valid_f1c',
            'valid_h',
            'valid_nmi']}

        for item in self.metrics:
            output['iteration'].append(item['iteration'])
            for f in self._metric_names:
                output['train_{}'.format(f)].append(item['train'][f])
                output['valid_{}'.format(f)].append(item['valid'][f])

        return output

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)


class MultitaskDEC(DEC):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metrics = None
        self.n_classes = n_classes

    def _calculate_metrics(self, x, y, y_pred, c_map):
        cluster_pred = self.model.predict(x, verbose=0)[1].argmax(1)
        f1 = f1_score(y[:,1], np.argmax(y_pred, axis=1))
        f1c = calc_f1_score(y[:,1], cluster_pred, c_map)
        h = homogeneity_score(y[:,1], cluster_pred)
        nmi = metrics.normalized_mutual_info_score(y[:,1], cluster_pred)

        return (f1, f1c, h, nmi)

    def build_model(self, alpha, beta, gamma, loss, loss_weights):
        cluster_weights = self.model.get_layer(name='clustering').get_weights()

        a = Input(shape=(self.dims[0],))  # input layer

        self.model.layers[1].kernel_regularizer = regularizers.l2(0.5)
        self.model.layers[2].kernel_regularizer = regularizers.l2(0.5)
        self.model.layers[3].kernel_regularizer = regularizers.l2(0.5)
        self.model.layers[4].kernel_regularizer = regularizers.l2(0.5)

        hidden = self.encoder(a)
        q_out = ClusteringLayer(self.n_clusters, name='clustering')(hidden)

        e_out = self.autoencoder(a)

        pred = Dense(2, activation='softmax')(q_out)

        self.model = Model(inputs=a, outputs=[pred, q_out, e_out])

        self.model.get_layer(name='clustering').set_weights(cluster_weights)

        optimizer = 'adam'

        if loss is None:
            self.model.compile(
                optimizer=optimizer,
                loss={
                    'dense_1': 'categorical_crossentropy',
                    'clustering': 'kld', 'model_1': 'mse'},
                loss_weights={
                    'dense_1': alpha, 'clustering': beta, 'model_1': gamma})
        else:
            self.model.compile(
                optimizer=optimizer,
                loss=loss,
                loss_weights=loss_weights)

    def evaluate(self):
        pass

    def clustering(
            self,
            x,
            y,
            train_dev_data,
            validation_data,
            tol=1e-3,
            update_interval=140,
            maxiter=2e4,
            save_dir='./results/dec',
            save_interval=5,
            pretrained_weights=None,
            alpha=K.variable(1.0),
            beta=K.variable(0.0),
            gamma=K.variable(0.0),
            loss_weight_decay=True,
            loss=None,
            loss_weights=None):

        if not os.path.isdir(save_dir):
            raise FileNotFoundError(
                'savedir does not exist\n{}'.format(save_dir))
        if y is None:
            logger.warn('No labels provided, won\'t print metrics')

        print('Update interval', update_interval)
        print('Save interval', save_interval)
   
        try:
            self.load_weights(pretrained_weights)
        except AttributeError:
            # initialize cluster centers using k-means
            print('Initializing cluster centers with k-means.')
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(self.encoder.predict(x))
            y_pred_last = y_pred
            self.model.get_layer(name='clustering') \
                .set_weights([kmeans.cluster_centers_])
   
        y_p = self.predict_clusters(x)
   
        cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
            get_cluster_to_label_mapping_safe(
                y[:,1], y_p, self.n_classes, self.n_clusters)
        
        logger.debug('')
        print(np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list)))
        cluster_to_label_mapping[np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list))] = 1
        
        ###############################################################
        ###############################################################
        ###############################################################

        logger.debug('Building Model')
        self.build_model(alpha, beta, gamma, loss, loss_weights)

        if not os.path.isdir(save_dir):
            logger.debug('Save dir doesn\'t exist')
            os.makedirs(save_dir)
   
        loss = [0, 0, 0]
        index = 0
        q = self.model.predict(x, verbose=0)[1]
        y_pred_last = q.argmax(1)
        self.metrics = Metrics()

        best_train_dev_loss = [np.inf, np.inf, np.inf]
        logger.debug('start training')
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)[1]
                valid_p = self.target_distribution(self.model.predict(validation_data[0], verbose=0)[1])
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                y_pred = self.model.predict(x)[0]
                if y is not None:
                    logger.debug('Calculating metrics')
                    c_map = get_cluster_to_label_mapping_safe(
                        y[:,1], q.argmax(1), self.n_classes,
                        self.n_clusters, toprint=False)[0]

                    metrics_train = self._calculate_metrics(x, y, y_pred, c_map)

                    x_valid = validation_data[0]
                    y_valid = validation_data[1]
                    y_pred_valid = self.model.predict(validation_data[0])[0]

                    val_loss = self.model.test_on_batch(
                        x_valid, [y_valid, valid_p, x_valid])
                    metrics_valid = self._calculate_metrics(
                        x_valid, y_valid, y_pred_valid, c_map)
                    print(self.metrics.add(
                        ite, metrics_train, metrics_valid, loss, val_loss))
                    
                    train_dev_p = self.target_distribution(
                        self.model.predict(train_dev_data[0], verbose=0)[1])
                    train_dev_loss = np.round(self.model.test_on_batch(
                        train_dev_data[0],
                        [train_dev_data[1], train_dev_p, train_dev_data[0]]
                        ), 5)
                    if train_dev_loss[1] < best_train_dev_loss[1] and \
                            train_dev_loss[-1] < best_train_dev_loss[-1]:
                            # only interested in classification improvements
                    
                        print('saving model: {} -> {}'.format(
                            best_train_dev_loss, train_dev_loss))
                        print('saving model: ', best_train_dev_loss, ' -> ', train_dev_loss)
                        self.model.save_weights(os.path.join(
                            save_dir, 'best_train_dev_loss.hf'))
                        best_train_dev_loss = train_dev_loss
                        best_ite = ite

                # check stop criterion
                
                if ite > 0 and delta_label < tol:
                    print('delta_label {} < tol {}'.format(delta_label, tol))
                    print('Reached tolerance threshold. Stopping training.')
                    break
                
                # Classification loss
                alpha = K.variable((1 - ite/maxiter))
                # Clustering loss
                beta  = K.variable(1-alpha)  # should ignore l=this loss
                # reconstruction loss
                gamma = K.variable(1.0)
                print(K.eval(alpha), K.eval(beta), K.eval(gamma))
                history = self.model.fit(
                    x=x,
                    y=[y,p,x],
                    validation_data=(
                        validation_data[0], [validation_data[1], valid_p,
                        validation_data[0]]),
                    callbacks=[MyLossWeightCallback(alpha, beta, gamma)],
                    verbose=0)
            else:
                print(K.eval(alpha), K.eval(beta), K.eval(gamma))
                history = self.model.fit(
                    x=x, y=[y,p,x],
                    validation_data=(
                        validation_data[0], [validation_data[1], valid_p,
                        validation_data[0]]),
                    verbose=0)
            #history = self.model.fit(x=x, y=[y,p], callbacks=[MyLossWeightCallback(alpha, beta)], verbose=0)
            #print(history.history)
            loss = [history.history[k][0] for k in history.history.keys() if 'val' not in k]
              # save intermediate model
            if ite % save_interval == 0:
                # save IDEC model checkpoints
                fname = os.path.join(save_dir, 'DEC_model_{}.h5'.format(ite))
                print('saving model to: {}'.format(fname))
                self.model.save_weights(fname)

            self.metrics.save(os.path.join(
                save_dir, 'metrics_intermediate.pkl'))

            ite += 1

        # save the trained model
        fname = os.path.join(save_dir, 'DEC_model_final.h5')
        print('saving model to: {}'.format(fname))
        self.model.save_weights(fname)

        y_p = self.model.predict(x, verbose=0)[1].argmax(1)
        cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
            get_cluster_to_label_mapping_safe(
                y[:,1], y_p, self.n_classes, self.n_clusters)
        return y_pred, self.metrics, best_ite


