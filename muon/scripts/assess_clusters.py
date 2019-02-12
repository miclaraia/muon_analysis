import click
import pandas
import numpy as np

from muon.subjects.database import Database
from muon.project.clustering import Clustering
from muon.subjects.storage import Storage


@click.group(invoke_without_command=True)
@click.argument('database_file')
def main(database_file):
    database = Database(database_file)
    subject_storage = Storage(database)
    
    clusters = Clustering.cluster_assignment_counts(subject_storage, 'vegas')
    data = np.array([
        clusters[:,0],
        clusters[:,1],
        clusters[:,1]/np.sum(clusters, axis=1)
        ]).T
    data = pandas.DataFrame(data, columns=['non-muon', 'muon', '% muon'])
    print(data)

    import code
    code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
