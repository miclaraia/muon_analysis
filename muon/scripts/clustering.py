
import muon.utils.clustering as clustering
import swap.config

def download(cluster, x, y, c, size, prefix, dir_):
    subjects = cluster.subjects_in_range(x, y, c)
    cluster.download_subjects(subjects, size, prefix, dir_)

def regions(path):
    swap.config.logger.init()
    subjects = clustering.Subjects(path)
    cluster = clustering.Cluster.create(subjects)

    download(cluster, (-1.21084, 2.60831), (-1.87769, 1.00803), None,
             100, 'red_region', './middle')
    download(cluster, (-20, -1.3), None, None,
             100, 'blue_region', './left')
    download(cluster, (-20, -1.3), None, None,
             100, 'outer_region', './right')





