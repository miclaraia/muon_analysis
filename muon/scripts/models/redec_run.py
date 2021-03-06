import pickle
import numpy as np
import os
import click
import shutil
from datetime import datetime


from muon.dissolving.redec import ReDEC, Config
from muon.dissolving.multitask import MultitaskDEC
from muon.dissolving.multitask import Config as MultitaskConfig
import muon.deep_clustering.clustering


def data_path(*args):
    return os.path.join(os.getenv('MUOND'), *args)


def load_metrics(save_dir):
    with open(os.path.join(save_dir, 'results_final.pkl'), 'rb') as f:
        return pickle.load(f)['metrics']


@click.group(invoke_without_command=True)
@click.option('--source_dir', required=True)
@click.option('--save_dir')
@click.option('--model_name', required=True)
@click.option('--name', required=True)
@click.option('--batch_size', default=256, type=int)
@click.option('--lr', default=0.01, type=float)
@click.option('--momentum', default=0.9, type=float)
@click.option('--tol', default=0.001, type=float)
@click.option('--epochs', default=80, type=int)
@click.option('--save_interval', default=5, type=int)
@click.option('--update_interval', default=140, type=int)
def main(
        source_dir,
        save_dir,
        model_name,
        name,
        batch_size,
        lr,
        momentum,
        tol,
        epochs,
        save_interval,
        update_interval):

    if not save_dir:
        model_name = '{}-{}'.format(
            model_name, datetime.now().isoformat(timespec='seconds'))
        save_dir = os.path.join(
            os.getenv('MUOND'), 'clustering_models', model_name)
    if os.path.isdir(save_dir):
        raise FileExistsError(save_dir)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(source_dir)
    os.makedirs(save_dir)

    source_config = MultitaskConfig.load(os.path.join(
        source_dir, 'config.json'))

    splits_file = source_config.splits_file

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
        'source_dir': source_dir,
        'splits_file': splits_file,
        'n_classes': 2,
        'n_clusters': source_config.n_clusters,
        'update_inteval': update_interval,
        'nodes': source_config.nodes,
        'batch_size': batch_size,
        'optimizer': ('SGD', {'lr': lr, 'momentum': momentum}),
        'tol': tol,
        'maxiter': epochs,
        'save_interval': save_interval,
    }

    ae_weights = os.path.join(source_dir, 'ae_weights.h5')
    dec_weights = os.path.join(source_dir, 'best_train_dev_loss.h5')
    config_args['source_weights'] = ae_weights, dec_weights

    ae_weights = os.path.join(save_dir, 'ae_weights.h5')
    dec_weights = os.path.join(save_dir, 'DEC_model_final.h5')
    config_args['save_weights'] = ae_weights, dec_weights

    config = Config(**config_args)
    config.dump()
    for i in range(2):
        shutil.copyfile(config.source_weights[i], config.save_weights[i])

    mdec = MultitaskDEC.load(source_dir, x_train)

    redec = ReDEC(config, x_train.shape)
    redec.init(x_train)
    redec.load_multitask_weights(mdec)

    y_pred, metrics = redec.clustering(
        (x_train, y_train),
        (x_train_dev, y_train_dev),
        (x_test, y_test),
        (x_valid, y_valid))

    with open(os.path.join(save_dir, 'results_final.pkl'), 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'metrics': metrics}, f)

    redec.report_run(splits)

if __name__ == '__main__':
    main()
