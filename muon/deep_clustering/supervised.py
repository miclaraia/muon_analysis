
import os
import numpy as np
import logging
import json
import csv

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.metrics import f1_score

from muon.utils.camera import CameraRotate
import muon.data

logger = logging.getLogger(__name__)


class Supervised:

    def __init__(self, config):
        self.config = config
        self.subsets = self._subsets()

        self.model = self.init_model()

    def _subsets(self):
        fname = {'train': 'train.csv', 'validate': 'test.csv'}
        def load(fname):
            fname = os.path.join(self.config.save_dir, fname)
            with open(fname, 'r') as file:
                reader = csv.DictReader(file)
                return [int(item['subject']) for item in reader]
        return {k:load(fname[k]) for k in fname}

    def init_model(self):
        input_shape = (None, self.config.input_shape[1])
        optimizer = self.config.optimizer
        loss = self.config.loss

        model = Sequential()
        model.add(Dense(500, activation='relu', input_dim=499))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(2000, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss=loss, optimizer=optimizer)

        ae_weights = self.config.ae_weights
        if os.path.isfile(ae_weights):
            model.load_weights(ae_weights)

        return model

    def save(self):
        self.model.save_weights(self.config.ae_weights, True)

    @classmethod
    def load(cls, config):
        model = cls(config)
        model.model.load_weights(config.ae_weights)
        return model

    def score(self, subjects):
        rotation = self.config.rotation
        xy = self.get_xy_subsets(subjects, rotation)

        logger.debug('xy: %s', xy)

        scores = {}
        for k in xy:
            y_prob = self.model.predict(xy[k][0])
            y_pred= y_prob.argmax(axis=-1)
            logger.debug('%s y_prob: %s\ny_pred: %s', k, y_prob, y_pred)
            scores[k] = f1_score(xy[k][1], y_pred)

        return scores

    @staticmethod
    def get_xy(subjects, rotation, hot_encode=False):
        x = []
        y = []
        cr = CameraRotate()
        for s in subjects.iter():
            if rotation:
                for n in range(6):
                    x.append(cr.rotate(s.scaled_charge, n))
                    y.append(s.label)
            else:
                x.append(s.scaled_charge)
                y.append(s.label)

        x = np.array(x)
        y = np.array(y)
        if hot_encode:
            y = np_utils.to_categorical(y, 2)
        return x, y

    def get_xy_subsets(self, subjects, rotation, hot_encode=False):
        data = {}
        for k in self.subsets:
            subset = subjects.subset(self.subsets[k])
            data[k] = self.get_xy(subset, rotation, hot_encode)
        return data


    def train(self, subjects):
        rotation = self.config.rotation
        epochs = self.config.epochs

        xy = self.get_xy_subsets(subjects, rotation, hot_encode=True)
        train = xy['train']
        validate = xy['validate']

        callbacks = [
            ModelCheckpoint(
                self.config.checkpoint, monitor='val_loss',
                save_best_only=True, mode='min')
            ]

        model = self.model
        model.fit(*train, validation_data=validate, epochs=epochs,
                  batch_size=256, callbacks=callbacks)


class Config:
    def __init__(self, input_shape, save_dir, loss=None, optimizer=None,
                 rotation=False, **kwargs):
        self.loss = loss or 'categorical_crossentropy'
        self.optimizer = optimizer or 'adam'
        self.input_shape = input_shape
        self.rotation = rotation
        self.epochs = kwargs.get('epochs', 100)

        save_dir = save_dir or muon.data.dir()
        self.save_dir = os.path.abspath(save_dir)

        subjects = os.path.join(save_dir, 'subjects.pkl')
        subjects = kwargs.get('subjects', subjects)
        self.subjects = os.path.abspath(subjects)

        ae_weights = kwargs.get('ae_weights', None)
        if ae_weights is None:
            ae_weights = os.path.join(save_dir, 'ae_weights.h5')
        self.ae_weights = os.path.abspath(ae_weights)

        self.checkpoint = kwargs.get('checkpoint', None) or \
                os.path.join(save_dir, 'checkpoint_model.h5')
        
    def dump(self):
        fname = os.path.join(self.save_dir, 'config.json')
        json.dump(self.__dict__, open(fname, 'w'))

    @classmethod
    def load(cls, fname):
        data = json.load(open(fname, 'r'))
        config = cls(None, None)
        config.__dict__.update(data)
        return config

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)
