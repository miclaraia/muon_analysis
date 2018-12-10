import pickle
import os
import shutil

from keras import backend as K

from muon.dissolving.decv2 import DECv2
from muon.dissolving.multitask import MultitaskDEC
from muon.dissolving.redec import ReDEC
from muon.dissolving.utils import Config

runs = [#('mnt/clustering_models/original_dec', 'original dec'),
        #('mnt/clustering_models/aws/decv2/norotation/dec_1-2018-12-09T03:56:15', 'norotation dec 50 clusters'),
        # ('mnt/clustering_models/aws/decv2/norotation/multitask_1-2018-12-09T06:54:30', 'norotation multitask 50 clusters'),
        # ('mnt/clustering_models/aws/decv2/norotation/redec_1-2018-12-09T08:13:14', 'norotation redec 50 clusters'),
        # ('mnt/clustering_models/aws/decv2/norotation/dec_2-2018-12-09T06:25:30', 'norotation dec 10 clusters'),
        # ('mnt/clustering_models/aws/decv2/norotation/multitask_1-2018-12-09T10:38:26', 'norotation multitask 10 clusters'),
        # ('mnt/clustering_models/aws/decv2/norotation/redec_1-2018-12-09T17:20:11/', 'norotation redec 10 clusters'),
        # ('mnt/clustering_models/aws/decv2/norotation/dec_1-2018-12-09T08:29:41', 'norotation dec 10 clusters no y'),
        # ('mnt/clustering_models/aws/decv2/norotation/multitask_1-2018-12-09T09:52:20', 'norotation multitask 10 clusters no y'),
        # ('mnt/clustering_models/aws/decv2/norotation/redec_1-2018-12-09T17:17:21', 'norotation redec 10 clusters no y'),
        #('mnt/clustering_models/aws/decv2/norotation/dec_1-2018-12-09T17:58:33', 'norotation dec 50 clusters no y'),
        # ('mnt/clustering_models/aws/decv2/norotation/multitask_1-2018-12-09T21:54:47/', 'norotation multitask 50 clusters no y')]
        #('mnt/clustering_models/aws/decv2/rotation/dec_1-2018-12-09T18:36:16', 'rotation dec 50 clusters'),
        ('mnt/clustering_models/aws/decv2/norotation/redec_1-2018-12-09T22:21:22', 'norotation redec 50 clusters no y'),]


def main():
    for save_dir, name in runs:
        print(save_dir, name)

        save_dir = '/'.join(save_dir.split('/')[1:])
        save_dir = os.path.join(os.getenv('MUOND'), save_dir)
        print(save_dir)

        config = Config.load(os.path.join(save_dir, 'config.json'))
        with open(config.splits_file, 'rb') as f:
            splits = pickle.load(f)
            x_train = splits['train'][0]

        if config.name is None:
            fname = os.path.join(save_dir, 'metrics_intermediate.pkl')
            if os.path.isfile(fname):
                print('replacing metrics')
                print(fname)
                shutil.copyfile(fname, os.path.join(save_dir, 'metrics.pkl'))
                os.remove(fname)

            if 'multitask' in save_dir:
                fname = os.path.join(save_dir, 'best_train_dev_loss.h5')
                print('replacing best_train_dev', fname)
                shutil.copyfile(fname, config.save_weights[1])

        K.clear_session()
        if 'multitask' in save_dir:
            print('MultitaskDEC')
            dec = MultitaskDEC.load(save_dir, x_train, verbose=False)
        elif 'redec' in save_dir:
            print('ReDEC')
            dec = ReDEC.load(save_dir, x_train, verbose=False)
        else:
            print('DECv2')
            dec = DECv2.load(save_dir, x_train, verbose=False)

        dec.config.name = name
        print('running report')
        dec.report_run(splits)
        print('dump config')
        dec.config.dump()

if __name__ == '__main__':
    main()


