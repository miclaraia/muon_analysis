
import os

from muon.dissolving.multitask import Config
import muon.deep_clustering.clustering as dcl

SPLITS_FILES = {
    'hugh': 'tt_split_hugh_xy.pkl',
    'volunteer_majority': 'tt_split_volunteer_majority_all_xy.pkl'
}


def model_path(path):
    return os.path.join(os.getenv('MUOND'), 'clustering_models', path)


def splits_file(name):
    return os.path.join(os.getenv('MUOND'), 'subjects', SPLITS_FILES[name])


def make_config(save_dir, source_dir, **kwargs):
    ae_weights = os.path.join(source_dir, 'ae_weights.h5')
    dec_weights = os.path.join(source_dir, 'DEC_model_final.h5')
    kwargs['source_weights'] = ae_weights, dec_weights

    ae_weights = os.path.join(save_dir, 'ae_weights.h5')
    dec_weights = os.path.join(save_dir, 'DEC_model_final.h5')
    kwargs['save_weights'] = ae_weights, dec_weights

    kwargs['save_dir'] = save_dir

    return Config(**kwargs)


def main():
    source_dir = os.path.join(
        os.getenv('MUOND'),
        'clustering_models/dec/dec_no_labels')
    source_config = dcl.Config.load(os.path.join(source_dir, 'config.json'))
    make_config(
        save_dir=model_path('run_multitask_1'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        batch_size=256,
        optimizer=('SGD', {'lr': 0.01, 'momentum': 0.9}),
        tol=0.001,
        maxiter=10,
        update_interval=1,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
        alpha=1.0,
        beta=0.0,
        gamma=0.0
    ).dump()

    make_config(
        save_dir=model_path('run_multitask_2'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        batch_size=256,
        optimizer=('SGD', {'lr': 0.01, 'momentum': 0.9}),
        tol=0.001,
        maxiter=20,
        update_interval=1,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
        alpha=1.0,
        beta=0.0,
        gamma=0.0
    ).dump()

    make_config(
        save_dir=model_path('run_multitask_3'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        batch_size=256,
        optimizer=('SGD', {'lr': 0.01, 'momentum': 0.9}),
        tol=0.001,
        maxiter=20,
        update_interval=1,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
        alpha=1.0,
        beta=0.0,
        gamma=1.0
    ).dump()

    make_config(
        save_dir=model_path('run_multitask_4'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        batch_size=256,
        optimizer=('SGD', {'lr': 0.01, 'momentum': 0.9}),
        tol=0.001,
        maxiter=80,
        update_interval=1,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
        alpha=1.0,
        beta=0.0,
        gamma=1.0
    ).dump()

    make_config(
        save_dir=model_path('run_multitask_5'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        batch_size=256,
        optimizer=('SGD', {'lr': 0.01, 'momentum': 0.9}),
        tol=0.001,
        maxiter=80,
        update_interval=1,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
        alpha=1.0,
        beta=0.0,
        gamma=0.5
    ).dump()

    make_config(
        save_dir=model_path('run_multitask_6'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        batch_size=256,
        optimizer=('SGD', {'lr': 0.01, 'momentum': 0.9}),
        tol=0.001,
        maxiter=80,
        update_interval=1,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
        alpha=1.0,
        beta=0.0,
        gamma=0.25
    ).dump()

    make_config(
        save_dir=model_path('run_multitask_6'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        batch_size=256,
        optimizer=('SGD', {'lr': 0.01, 'momentum': 0.9}),
        tol=0.001,
        maxiter=80,
        update_interval=1,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
        alpha=1.0,
        beta=0.0,
        gamma=0.25
    ).dump()

    make_config(
        save_dir=model_path('aws/run_volunteer_1'),
        source_dir=source_dir,
        splits_file=splits_file('volunteer_majority'),
        n_clusters=source_config.n_clusters,
        nodes=source_config.nodes,
        maxiter=80,
        gamma=1.0
    ).dump()


if __name__ == '__main__':
    main()
    
    
