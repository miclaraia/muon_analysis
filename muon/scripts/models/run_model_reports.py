import pickle
import os
import shutil
import csv

from keras import backend as K

from muon.dissolving.decv2 import DECv2
from muon.dissolving.multitask import MultitaskDEC
from muon.dissolving.redec import ReDEC
from muon.dissolving.utils import Config


def main():
    fname = os.path.join(
        os.getenv('MUON'), 'muon/scripts/models/decv2/models.csv')
    with open(fname) as f:
        reader = csv.reader(f)
        next(reader)
        runs = [i[0].strip() for i in reader if int(i[2])]

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
        else:
            print('DECv2')
            dec = DECv2.load(save_dir, x_train, verbose=False)

        print('running report')
        dec.report_run(splits)

