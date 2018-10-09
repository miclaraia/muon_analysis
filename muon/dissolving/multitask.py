import numpy as np
import os

from sklearn.metrics import f1_score, roc_curve
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import homogeneity_score

from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Dropout
from keras.initializers import Initializer
from keras.optimizers import SGD, Adadelta
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from keras.engine.topology import Layer
from keras import backend as K
from keras import regularizers
# from keras.models import load_model

from dec_keras.DEC import DEC, ClusteringLayer, cluster_acc

lcolours = ['#D6FF79', '#B0FF92', '#A09BE7', '#5F00BA', '#56CBF9', \
            '#F3C969', '#ED254E', '#CAA8F5', '#D9F0FF', '#46351D']

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
    _metric_names = ['fom', 'f1', 'f1c', 'h', 'nmi']

    def __init__(self):
        self.metrics = []

    def add(self, iteration, train, valid, loss, vloss):
        metrics = {
            'iteration': iteration,
            'train': {f: train[i] for i, f in enumerate(self._metric_names)},
            'valid': {f: valid[i] for i, f in enumerate(self._metric_names)},
        }
        self.metrics.append(metrics)

        s = '%4d 1%fpr=%.4f  F1=%.4f  F1c=%.4f  h=%.4f  nmi=%.f4\n' \
            '     v1%fpr=%.4f vF1=%.4f vF1c=%.4f vh=%.4f vnmi=%.4f\n' \
            '     loss=%s vloss=%s\n'

        m = (len(self.metrics),
             *(metrics['train'][f] for f in self._metric_names),
             *(metrics['valid'][f] for f in self._metric_names),
             loss, vloss)
        return s % m

    def dump(self):
        output = {k: [] for k in [
            'iteration',
            'train_fom',
            'train_f1',
            'train_f1c',
            'train_h',
            'train_nmi',
            'valid_fom',
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

        a = Input(shape=(400,))  # input layer

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
            y=None,
            train_dev_data=None,
            validation_data=None,
            tol=1e-3,
            update_interval=140,
            maxiter=2e4,
            save_dir='./results/dec',
            pretrained_weights=None,
            alpha=K.variable(1.0),
            beta=K.variable(0.0),
            gamma=K.variable(0.0),
            loss_weight_decay=True,
            loss=None,
            loss_weights=None):
        print('Update interval', update_interval)
        save_interval = x.shape[0] / self.batch_size * 5  # 5 epochs
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
        
        print(np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list)))
        cluster_to_label_mapping[np.argmax((1-np.array(majority_class_fractions))*np.array(n_assigned_list))] = 1
        
        ###############################################################
        ###############################################################
        ###############################################################

        self.build_model(alpha, beta, gamma, loss, loss_weights)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
   
        loss = [0, 0, 0]
        index = 0
        q = self.model.predict(x, verbose=0)[1]
        y_pred_last = q.argmax(1)
        self.metrics = Metrics()

        best_train_dev_loss = [np.inf, np.inf, np.inf]
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
                    c_map = get_cluster_to_label_mapping_safe(
                        y[:,1], q.argmax(1), self.n_classes,
                        self.n_clusters, toprint=False)[0]
                    f = one_percent_fpr(y[:,1], y_pred[:,1], 0.01)[0]

                    metrics_train = self._calculate_metrics(x, y, y_pred, c_map)

                    x_valid = validation_data[0]
                    y_valid = validation_data[1]
                    y_pred_valid = self.model.predict(validation_data[0])[0]

                    val_loss = self.model.test_on_batch(
                        x_valid, [y_valid, valid_p, x_valid])
                    metrics_valid = self._calculate_metrics(
                        x_valid, y_valid, y_pred_valid, c_map)
                    self.metrics.add(
                        ite, metrics_train, metrics_valid, loss, val_loss)
                    
                    train_dev_p = self.target_distribution(self.model.predict(train_dev_data[0], verbose=0)[1])
                    train_dev_loss = np.round(self.model.test_on_batch(train_dev_data[0], [train_dev_data[1], train_dev_p, train_dev_data[0]]), 5)
                    if train_dev_loss[1] < best_train_dev_loss[1] and train_dev_loss[-1] < best_train_dev_loss[-1]: # only interested in classification improvements
                      print('saving model: ', best_train_dev_loss, ' -> ', train_dev_loss)
                      self.model.save_weights('best_train_dev_loss.hf')
                      best_train_dev_loss = train_dev_loss
                      best_ite = ite
            
                # check stop criterion
                
                if ite > 0 and delta_label < tol:
                    print('delta_label {} < tol {}'.format(delta_label, tol))
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break
                
                # train on batch
                """
                if (index + 1) * self.batch_size > x.shape[0]:
                  loss = self.model.train_on_batch(x=x[index * self.batch_size::],
                                                   y=[y[index * self.batch_size::], \
                                                      p[index * self.batch_size::]])
                  index = 0
                else:
                  loss = self.model.train_on_batch(x=x[index * self.batch_size:(index + 1) * self.batch_size],
                                                   y=[y[index * self.batch_size:(index + 1) * self.batch_size], \
                                                      p[index * self.batch_size:(index + 1) * self.batch_size]])
                  index += 1
                """
            
            if loss_weight_decay:
                """
                if ite < 50:
                  alpha = K.variable(1.0)
                  beta  = K.variable(0.0)
                  gamma = K.variable(1.0)
                elif ite >= 50:
                  #alpha = K.variable(1.0)
                  alpha = K.variable(0.0)
                  #beta  = K.variable(0.0)
                  beta  = K.variable(1.0)
                  gamma  = K.variable(1.0)
                  update_interval = 140
                  self.model.optimizer = SGD(lr=0.01, momentum=0.9)
                """
                """
                elif ite >= 200 and ite < 300:
                  #alpha = K.variable(1.0*(1 - ((ite - 200)/100.)))
                  alpha = K.variable(1.0)
                  beta  = K.variable(1.0)
                  gamma = K.variable(1.0)
                print(K.eval(alpha), K.eval(beta))
                """
                #alpha = K.variable(1.0*(1 - ((ite - 200)/100.)))
                """
                if ite < 40:
                  alpha = K.variable((1 - ite/maxiter))
                  beta  = K.variable(1-alpha)
                  gamma = K.variable(1.0)
                print(K.eval(alpha), K.eval(beta), K.eval(gamma))
                if ite == 40:
                  print('Initializing cluster centers with k-means.')
                  kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
                  y_pred = kmeans.fit_predict(self.encoder.predict(x))
                  self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
                if ite >= 40:
                  alpha = K.variable(0.0)
                  beta  = K.variable(1.0)
                  gamma = K.variable(1.0)
                  update_interval=140
                  self.model.optimizer = SGD(lr=0.01, momentum=0.9)
                """
                
                alpha = K.variable((1 - ite/maxiter))
                beta  = K.variable(1-alpha)
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

            ite += 1

        # save the trained model
        #logfile.close()
        fname = os.path.join(save_dir, 'DEC_model_final.h5')
        print('saving model to: {}'.format(fname))
        self.model.save_weights(fname)

        y_p = self.model.predict(x, verbose=0)[1].argmax(1)
        cluster_to_label_mapping, n_assigned_list, majority_class_fractions = \
            get_cluster_to_label_mapping_safe(
                y[:,1], y_p, self.n_classes, self.n_clusters)
        return y_pred, self.metrics.dump(), best_ite


def one_percent_fpr(y, pred, fom):
    fpr, tpr, thresholds = roc_curve(y, pred)
    FoM = 1-tpr[np.where(fpr<=fom)[0][-1]] # MDR at 1% FPR
    threshold = thresholds[np.where(fpr<=fom)[0][-1]]
    return FoM, threshold, fpr, tpr


def calc_f1_score(y_true, predicted_clusters, cluster_to_label_mapping):
    y_pred = []
    for i in range(len(y_true)):
          y_pred.append(cluster_to_label_mapping[predicted_clusters[i]])
    return f1_score(y_true, np.array(y_pred))

def get_cluster_to_label_mapping_safe(y, y_pred, n_classes, n_clusters, toprint=True):
    """Enusre at least one cluster assigned to each label.
    """
    one_hot_encoded = np_utils.to_categorical(y, n_classes)

    cluster_to_label_mapping = []
    n_assigned_list = []
    majority_class_fractions = []
    majority_class_pred = np.zeros(y.shape)
    for cluster in range(n_clusters):
        cluster_indices = np.where(y_pred == cluster)[0]
        n_assigned_examples = cluster_indices.shape[0]
        n_assigned_list.append(n_assigned_examples)
        cluster_labels = one_hot_encoded[cluster_indices]
        cluster_label_fractions = np.mean(cluster_labels, axis=0)
        majority_cluster_class = np.argmax(cluster_label_fractions)
        cluster_to_label_mapping.append(majority_cluster_class)
        majority_class_pred[cluster_indices] += majority_cluster_class
        majority_class_fractions.append(cluster_label_fractions[majority_cluster_class])
        if toprint:
            print(cluster, n_assigned_examples, majority_cluster_class, cluster_label_fractions[majority_cluster_class])
    #print(cluster_to_label_mapping)
    if toprint:
        print(np.unique(y), np.unique(cluster_to_label_mapping))
    try:
        # make sure there is at least 1 cluster representing each class
        assert np.all(np.unique(y) == np.unique(cluster_to_label_mapping))
    except AssertionError:
        # if there is no cluster for a class then we will assign a cluster to that
        # class

        # find which class it is
        # ASSUMPTION - this task is binary

        diff = list(set(np.unique(y)) - set(np.unique(cluster_to_label_mapping)))[0]
          # we choose the cluster that contains the most examples of the class with no cluster

        one_hot = np_utils.to_categorical(y_pred[np.where(y==diff)[0]], \
                                            len(cluster_to_label_mapping))

        cluster_to_label_mapping[np.argmax(np.sum(one_hot, axis=0))] = int(diff)
    if toprint:
        print(cluster_to_label_mapping)
    return cluster_to_label_mapping, n_assigned_list, majority_class_fractions
