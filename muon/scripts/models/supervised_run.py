import pickle
import numpy as np
import os
import click
import shutil
from datetime import datetime

from muon.dissolving.supervised import Supervised, Config


def data_path(*args):
    return os.path.join(os.getenv('MUOND'), *args)

@click.group(invoke_without_command=True)
@click.option('--splits_file', required=True)
@click.option('--save_dir')
@click.option('--model_name', required=True)
@click.option('--name', required=True)
@click.option('--batch_size', type=int)
@click.option('--lr', type=float, default=0.01)
@click.option('--momentum', type=float, default=0.9)
@click.option('--epochs', type=int)
def main(
        splits_file,
        save_dir, 
        model_name,
        batch_size,
        lr,
        momentum,
        epochs,
        name):

    if not save_dir:
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
        'name': name,
        'splits_file': splits_file,
        'n_classes': 2,
        'batch_size': batch_size,
        'optimizer': ('SGD', {'lr': lr, 'momentum': momentum}),
        'maxiter': epochs,
        'source_dir': None,
        'source_weights': (None, None),
        'save_weights': os.path.join(save_dir, 'model_weights_final.h5')
    }

    config = Config(**config_args)
    config.dump()

    dec = Supervised(config, x_train.shape)
    dec.init()

    y_pred = dec.train(
        (x_train, y_train),
        (x_train_dev, y_train_dev),
        (x_valid, y_valid))

    dec.report_run(splits)


if __name__ == '__main__':
    main()

