#!/usr/bin/env python

import os
import click
import pickle

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers

from muon.deep_clustering.clustering import Config, Cluster
from muon.deep_clustering.mapping2 import Mapping
from muon.deep_clustering.dec_updating import DEC_Updater
import muon.deep_clustering.utils as utils
import muon.project.parse_export as pe

import logging
logger = logging.getLogger(__name__)

_config = 'mnt/dec/dec_no_labels/config_jupyter.json'
train = utils.load_set('mnt/training_set/train.csv')
validate = utils.load_set('mnt/training_set/test.csv')


@click.group(invoke_without_command=True)
@click.argument('path')
@click.argument('name')
@click.argument('lr')
def main(path, name, lr):
    config = Config.load(_config)
    print(config.__dict__)
    subjects = pickle.load(open(config.subjects, 'rb'))

    labels = pe.Aggregate.load('mh2')
    truth = pe.Aggregate.load('mh2_gold')
    _labels = labels.subject_labels(), truth.subject_labels()

    subjects = subjects.subset(truth.labeled_subjects())
    truth.apply_labels(subjects)

    cluster = Cluster.create(subjects, config)
    cluster.initialize()
    mapping = DEC_Updater(cluster, *_labels, train, validate, 0.95)
    print(mapping.scores)

    optimizer = optimizers.SGD(lr=float(lr))
    mapping.init_model('categorical_crossentropy', optimizer)
    print(mapping.scores)

    weights_fname = os.path.join(path, name)
    kwargs = {
        'epochs': 500,
        'batch_size': 256,
        'callbacks': [
            ModelCheckpoint(
                weights_fname, monitor='val_loss',
                save_best_only=True, save_weights_only=True, mode='min'),
            EarlyStopping(
                monitor='val_loss', min_delta=0, patience=20,
                verbose=0, mode='min')]
    }

    def fit(model, train, val):
        model.fit(*train, validation_data=val, **kwargs)
    # def xy(train, subjects):
        # return subjects.get_xy_volunteer(labels.data['subjects'], False)
    mapping.apply_mapping(fit)

    print(mapping.cluster_mapping())
    print('scores:', utils.pd_scores(mapping.scores))


if __name__ == '__main__':
    main()
