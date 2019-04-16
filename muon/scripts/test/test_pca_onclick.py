save_dir = '/mnt/storage/science_data/muon_data/clustering_models/redec-2018-12-11T20:04:38'

# import matplotlib
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas
from keras import backend as K
import os
from IPython.display import Image
import csv

from muon.dissolving.redec import ReDEC
from muon.dissolving.redec import Config as RConfig
from keras import backend as K




config = RConfig.load(os.path.join(save_dir, 'config.json'))
with open(config.splits_file, 'rb') as file:
    splits = pickle.load(file)

dec = ReDEC.load(save_dir, splits['train'][0])
dec.report_run(splits)
        
# from muon.dissolving.utils import PCA_Plot

# x_train, y_train = splits['train']

# pcplot = PCA_Plot(dec, x_train, y_train, config.n_clusters,
                  # title='', make_samples=True)

# fig = pcplot.plot()
# plt.show()

# pcplot.plot_samples(save_dir)
# fig, pca, samples = pca_plotv2(dec, x_train, y_train, config.n_clusters, title='')

# fig.show()
# for l, sample_fig in samples:
    # fname = os.path.join(save_dir, 'sample_test_{}'.format(l))
    # #sample_fig.savefig(fname)
    # sample_fig.show()

# fig.show()

# import code
# code.interact(local={**globals(), **locals()})
