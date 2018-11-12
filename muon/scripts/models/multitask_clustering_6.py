import pickle
import numpy as np
import os
import click

import keras.backend as K
from keras.optimizers import SGD
from keras.utils import np_utils

from itertools import combinations_with_replacement

from muon.dissolving.multitask import MultitaskDEC
from muon.subjects.storage import Storage
from muon.deep_clustering.clustering import Cluster, Config

lcolours = ['#CAA8F5', '#D6FF79', '#A09BE7', '#5F00BA', '#56CBF9', \
            '#F3C969', '#ED254E', '#B0FF92', '#D9F0FF', '#46351D']

def data_path(path):
    return os.path.join(os.getenv('MUOND'), path)

@click.group(invoke_without_command=True)
@click.argument('splits_file')
@click.argument('save_dir')
def main(splits_file, save_dir):
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)

    x_test, y_test = splits['test']
    x_train, y_train = splits['train']
    x_valid, y_valid = splits['valid']
    x_train_dev, y_train_dev = splits['train_dev']

    order = np.random.permutation(x_train.shape[0])
    print(x_train.shape, y_train.shape)
    x_train = x_train[order,:]
    y_train = y_train[order]
    print(x_train.shape, x_test.shape, x_valid.shape, x_train_dev.shape)

    config = Config.load(data_path('clustering_models/dec/dec_no_labels/config.json'))
    ae_weights = os.path.join(save_dir, 'ae_weights.h5')
    dec_weights = os.path.join(save_dir, 'DEC_model_final.h5')
    # ae_weights = data_path('clustering_models/dec/dec_no_labels/ae_weights.h5')
    # dec_weights = data_path('clustering_models/dec/dec_no_labels/DEC_model_final.h5')

    # DEC constants from DEC paper
    batch_size = 256
    lr         = 0.01
    momentum   = 0.9
    tol        = 0.001
    maxiter    = 80
    # update_interval = 140 # perhaps this should be 1 for multitask learning
    update_interval = 1  # perhaps this should be 1 for multitask learning
    n_clusters = config.n_clusters  # number of clusters to use
    n_classes  = 2  # number of classes

    dims = [x_train.shape[1]] + config.nodes

    dec = MultitaskDEC(
        n_classes=n_classes,
        dims=dims,
        n_clusters=n_clusters,
        batch_size=batch_size)

    dec.initialize_model(
        optimizer=SGD(lr=lr, momentum=momentum),
        ae_weights=ae_weights,
        x=x_train)

    dec.model.load_weights(dec_weights)
    print(dec.model.summary())

    y_pred, metrics, best_ite = dec.clustering(
        x_train, np_utils.to_categorical(y_train),
        (x_train_dev, np_utils.to_categorical(y_train_dev)),
        (x_valid, np_utils.to_categorical(y_valid)),
        pretrained_weights=dec_weights,
        maxiter=maxiter,
        alpha=K.variable(1.0),
        beta=K.variable(0.0),
        gamma=K.variable(.25),
        loss_weight_decay=False,
        save_dir=save_dir,
        update_interval=update_interval)

    with open(os.path.join(save_dir, 'results_final.pkl'), 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'metrics': metrics,
            'best_ite': best_ite}, f)


if __name__ == '__main__':
    main()
