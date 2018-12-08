
import pickle
import numpy as np
import os
import click
import shutil
from datetime import datetime

from muon.dissolving.decv2 import DECv2, Config


def data_path(*args):
    return os.path.join(os.getenv('MUOND'), *args)

@click.group(invoke_without_command=True)
@click.option('--splits_file', required=True)
@click.option('--model_name', required=True)
@click.option('--batch_size', type=int)
@click.option('--lr', type=float, default=0.01)
@click.option('--momentum', type=float, default=0.9)
@click.option('--tol', type=float)
@click.option('--maxiter', type=int)
@click.option('--save_interval', type=int)
def main(
        splits_file,
        model_name,
        batch_size,
        lr,
        momentum,
        tol,
        maxiter,
        save_interval):

    model_name = '{}-{}'.format(
        model_name, datetime.now().replace(microsecond=0).isoformat())
    save_dir = os.path.join(
        os.getenv('MUOND'), 'clustering_models', model_name)
    if os.path.isdir(save_dir):
        raise FileExistsError(save_dir)
    os.makedirs(save_dir)

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

    config_args = {
        'save_dir': save_dir,
        'source_dir': None,
        'splits_file': splits_file,
        'n_classes': 2,
        'n_clusters': 50,
        'update_inteval': 1,
        'batch_size': batch_size,
        'optimizer': ('SGD', {'lr': lr, 'momentum': momentum}),
        'tol': tol,
        'maxiter': maxiter,
        'save_interval': save_interval,
        'source_weights': (None, None),
    }

    ae_weights = os.path.join(save_dir, 'ae_weights.h5')
    dec_weights = os.path.join(save_dir, 'DEC_model_final.h5')
    config_args['save_weights'] = ae_weights, dec_weights

    config = Config(**config_args)
    config.dump()


    dec = DECv2(config, x_train.shape)
    dec.init(x_train)

    y_pred = dec.clustering(
        (x_train, y_train),
        (x_train_dev, y_train_dev),
        (x_valid, y_valid))

    with open(os.path.join(save_dir, 'results_final.pkl'), 'wb') as f:
        pickle.dump({'y_pred': y_pred}, f)


if __name__ == '__main__':
    main()
