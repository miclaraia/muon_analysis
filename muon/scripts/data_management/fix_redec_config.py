
import os

from muon.dissolving.redec import Config
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

    print(kwargs)
    return Config(**kwargs)


def main():
    source_dir = model_path('run_multitask_4_1')
    source_config = dcl.Config.load(os.path.join(source_dir, 'config.json'))
    make_config(
        save_dir=model_path('run_multitask_4_1/run_redec_1'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        maxiter=10,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
    ).dump()

    source_dir = model_path('aws/run_volunteer_1')
    source_config = dcl.Config.load(os.path.join(source_dir, 'config.json'))
    make_config(
        save_dir=model_path('aws/run_volunteer_1/run_redec_1'),
        source_dir=source_dir,
        splits_file=splits_file('hugh'),
        maxiter=10,
        n_clusters=source_config.n_clusters,
        n_classes=2,
        nodes=source_config.nodes,
    ).dump()


if __name__ == '__main__':
    main()
    
    
