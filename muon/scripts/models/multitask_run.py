import pickle
import numpy as np
import os
import click
import shutil

from muon.dissolving.multitask import MultitaskDEC, Config
import muon.deep_clustering.clustering


def data_path(*args):
    return os.path.join(os.getenv('MUOND'), *args)

@click.group(invoke_without_command=True)
@click.option('--splits_file', required=True)
@click.option('--source_dir', required=True)
@click.option('--save_dir', required=True)
@click.option('--batch_size', type=int)
@click.option('--lr', type=float, default=0.01)
@click.option('--momentum', type=float, default=0.9)
@click.option('--tol', type=float)
@click.option('--maxiter', type=int)
@click.option('--save_interval', type=int)
@click.option('--alpha', type=float)
@click.option('--beta', type=float)
@click.option('--gamma', type=float)
def main(
        splits_file,
        source_dir,
        save_dir,
        batch_size,
        lr,
        momentum,
        tol,
        maxiter,
        save_interval,
        alpha, beta, gamma):
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

    source_config = muon.deep_clustering.clustering.Config.load(data_path(
        source_dir, 'config.json'))

    config_args = {
        'save_dir': save_dir,
        'splits_file': splits_file,
        'n_classes': 2,
        'n_clusters': source_config.n_clusters,
        'update_inteval': 1,
        'nodes': source_config.nodes,
        'batch_size': batch_size,
        'optimizer': ('SGD', {'lr': lr, 'momentum': momentum}),
        'tol': tol,
        'maxiter': maxiter,
        'save_interval': save_interval,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }

    ae_weights = os.path.join(source_dir, 'ae_weights.h5')
    dec_weights = os.path.join(source_dir, 'DEC_model_final.h5')
    config_args['source_weights'] = ae_weights, dec_weights

    ae_weights = os.path.join(save_dir, 'ae_weights.h5')
    dec_weights = os.path.join(save_dir, 'DEC_model_final.h5')
    config_args['save_weights'] = ae_weights, dec_weights

    config = Config(**config_args)
    config.dump()
    for i in range(2):
        shutil.copyfile(config.source_weights[i], config.save_weights[i])

    dec = MultitaskDEC(config, x_train.shape)
    dec.init(x_train)
    print(dec.model.summary())

    y_pred, metrics, best_ite = dec.clustering(
        (x_train, y_train),
        (x_train_dev, y_train_dev),
        (x_valid, y_valid))

    with open(os.path.join(save_dir, 'results_final.pkl'), 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'metrics': metrics,
            'best_ite': best_ite}, f)


if __name__ == '__main__':
    main()
