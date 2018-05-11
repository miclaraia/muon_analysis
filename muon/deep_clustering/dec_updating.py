
from keras.models import Model
from keras.layers import Input
from keras.utils import np_utils

from muon.deep_clustering.mapping2 import Mapping


class DEC_Updater(Mapping):

    def init_model(self, loss, optimizer):
        a = Input(shape=(self.width,)) # input layer
        pred = self.cluster.dec.model(a)
        model = Model(inputs=a, outputs=pred)
        model.compile(loss=loss, optimizer=optimizer)
        self.model = model

        self.score()
        return model

    def fit(self, fit_callback, xy_callback=None):
        if xy_callback:
            train = xy_callback(self.train, self.cluster.subjects)
        else:
            train = self.train.get_xy()
        val = self.validate.get_xy()

        def make_data(xy):
            x, y = xy
            n_clusters = self.cluster.config.n_clusters
            return (x, np_utils.to_categorical(y, n_clusters))

        val = make_data(val)
        train = make_data(train)

        fit_callback(self.model, train, val)
        self.score()
