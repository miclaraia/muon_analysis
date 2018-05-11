#!/usr/bin/env python

import os
import click
import pickle

from muon.utils.subjects import Subjects
from muon.deep_clustering.clustering import Config, Cluster
from muon.deep_clustering.mapping2 import Mapping
import muon.deep_clustering.utils as utils
import muon.project.parse_export as pe

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers

import logging
logger = logging.getLogger(__name__)

_config = 'mnt/dec/dec_no_labels/config_jupyter.json'
train = utils.load_set('mnt/training_set/train.csv')
validate = utils.load_set('mnt/training_set/test.csv')


@click.group(invoke_without_command=True)
@click.argument('path')
@click.argument('name')
def main(path, name):
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
    mapping = Mapping(cluster, *_labels, train, validate, 0.95)

    optimizer = optimizers.SGD(lr=0.5)
    mapping.init_model('categorical_crossentropy', optimizer)
    print(mapping.scores)

    kwargs = {
        'epochs': 500,
        'batch_size': 256,
        'callbacks': [
            ModelCheckpoint(
                utils.checkpoint_name(path), monitor='val_loss',
                save_best_only=True, mode='min'),
            EarlyStopping(
                monitor='val_loss', min_delta=0, patience=20,
                verbose=0, mode='min')]
    }

    def fit(model, train, val):
        model.fit(*train, validation_data=val, **kwargs)
        model.save_weights(os.path.join(path, name))
    def xy(train, subjects):
        return subjects.get_xy_volunteer(labels.data['subjects'], False)
    mapping.apply_mapping(fit, xy)

    print(mapping.cluster_mapping())
    print('scores:', utils.pd_scores(mapping.scores))


if __name__ == '__main__':
    main()
