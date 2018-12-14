import pickle
import os
import shutil
import csv
import click

from keras import backend as K

from muon.dissolving.decv2 import DECv2
from muon.dissolving.multitask import MultitaskDEC
from muon.dissolving.redec import ReDEC
from muon.dissolving.supervised import Supervised
from muon.dissolving.utils import Config, StandardMetrics


@click.group(invoke_without_command=True)
@click.argument('model')
@click.argument('standardmetrics')
def main(model, standardmetrics):
    fname = os.path.join(
        os.getenv('MUON'), 'muon/scripts/models/decv2/{}.csv'.format(model))
    with open(fname) as f:
        reader = csv.reader(f)
        next(reader)
        runs = []
        for row in reader:
            if int(row[2].strip()) and int(row[3].strip()):
                runs.append(row[0].strip())

    for save_dir in runs:
        config = Config.load(os.path.join(save_dir, 'config.json'))
        with open(config.splits_file, 'rb') as f:
            splits = pickle.load(f)
            x_train = splits['train'][0]

        K.clear_session()
        if 'multitask' in save_dir:
            print('MultitaskDEC')
            dec = MultitaskDEC.load(save_dir, x_train, verbose=False)
        elif 'redec' in save_dir:
            print('ReDEC')
            dec = ReDEC.load(save_dir, x_train, verbose=False)
        elif 'supervised' in save_dir:
            print('Supervised')
            dec = Supervised.load(save_dir, x_train, verbose=False)
        else:
            print('DECv2')
            dec = DECv2.load(save_dir, x_train, verbose=False)

        print('running report')
        dec.report_run(splits)

if __name__ == '__main__':
    main()
