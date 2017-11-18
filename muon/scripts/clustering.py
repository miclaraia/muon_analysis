
import muon.utils.clustering as clustering
import swap.config

import matplotlib.pyplot as plt

def download(cluster, x, y, c, size, prefix, dir_):
    subjects = cluster.subjects_in_range(x, y, c)
    cluster.download_subjects(subjects, size, prefix, dir_)

def regions(cluster, path):
    print('Middle region')
    download(cluster, (-1.21084, 3.0), (-1.87769, 1.00803), -1,
             200, 'red_region', './%s/middle' % path)
    print('Left region')
    download(cluster, (-20, -1.3), None, -1,
             200, 'blue_region', './%s/left' % path)
    print('Right region')
    download(cluster, (5, 100), None, -1,
             200, 'outer_region', './%s/right' % path)

def plot_classes(cluster, fname=None):
    fig = plt.figure()
    for i, s in enumerate((1, 0, -1)):
        cluster.plot_class(s, fig=fig, subplot=(2, 2, i+1), show=False,
                           save=fname)

    fig.show()


def all(cluster):
    subjects = cluster.subjects.list()
    cluster.plot_class([-1,0,1], True)





