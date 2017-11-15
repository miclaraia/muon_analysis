
import muon.utils.clustering as clustering
import swap.config

def download(cluster, x, y, c, size, prefix, dir_):
    subjects = cluster.subjects_in_range(x, y, c)
    cluster.download_subjects(subjects, size, prefix, dir_)

def regions(path):
    swap.config.logger.init()
    subjects = clustering.Subjects(path)
    cluster = clustering.Cluster.create(subjects)

    print('Middle region')
    download(cluster, (-1.21084, 3.0), (-1.87769, 1.00803), None,
             100, 'red_region', './middle')
    print('Left region')
    download(cluster, (-20, -1.3), None, None,
             100, 'blue_region', './left')
    print('Right region')
    download(cluster, (5, 100), None, None,
             100, 'outer_region', './right')





