
import os
from time import time

from keras.optimizers import SGD
import dec_keras as dk


# this is chosen based on prior knowledge of classes in the data set.
# n_clusters = 10 
# batch_size = 256
# # learning rate
# lr         = 0.01 
# momentum   = 0.9
# # tolerance - if clustering stops if less than this fraction
# # of the data changes cluster on an interation
# tol        = 0.001 

# maxiter         = 2e4
# update_interval = 140
# save_dir        = '../DEC-keras/results/'
# ae_weights = '../DEC-keras/results/mnist/ae_weights.h5'

class Config:
    def __init__(self, save_dir, ae_weights, **kwargs):
        self.n_clusters = kwargs.get('n_clusters', 10)
        self.batch_size = kwargs.get('batch_size', 256)
        self.nodes = kwargs.get('nodes', [500, 500, 2000, 10])
        self.lr = kwargs.get('lr', .01)
        self.momentum = kwargs.get('momentum', .9)
        self.tol = kwargs.get('tol', .001)
        self.maxiter = kwargs.get('maxiter', 2e4)
        self.update_interval = kwargs.get('update_interval', 140)
        self.save_dir = save_dir
        self.ae_weights = ae_weights

class Cluster:

    def __init__(self, dec, subjects, config):
        self.dec = dec
        self.subjects = subjects
        self.config = config

    @classmethod
    def create(cls, subjects, config):
        dec = dk.DEC(
            dims=[subjects.dimensions[1]] + config.nodes,
            n_clusters=config.n_clusters,
            batch_size=config.batch_size
        )

        

        dec.initialize_model(**{
            'optimizer': SGD(lr=config.lr, momentum=config.momentum),
            'ae_weights': config.ae_weights,
            'x': subjects.get_charge_array()
        })
        print(dec.model.summary())

        return cls(dec, subjects, config)

    def predict(self):
        order, charges = self.subjects.get_charge_array()
        labels = self.subjects.predictions(self, order)

        t0 = time()
        y_pred = self.dec.clustering(charges, y=labels, **{
            'tol': self.config.tol,
            'maxiter': self.config.maxiter,
            'update_interval': self.config.update_interval,
            'save_dir': self.config.save_dir
        })

        print('clustering time: %.2f' % time() - t0)

        accuracy = dk.cluster_acc(labels, y_pred)







