import click
import logging
import pickle

from muon.project.sql_images import SQLImages
from muon.subjects.storage import Storage

logger = logging.getLogger(__name__)


def load_cluster_assignments(cluster_assignment_file):
    with open(cluster_assignment_file, 'rb') as f:
        return pickle.load(f)


@click.group(invoke_without_command=True)
@click.argument('image_file')
@click.argument('subject_file')
@click.argument('cluster_assignment_file')
def main(image_file, subject_file, cluster_assignment_file):
    image_storage = SQLImages(image_file)
    subject_storage = Storage(subject_file)
    cluster_assignments = load_cluster_assignments(cluster_assignment_file)

    kwargs = {
        'image_size': 36,
        'image_width': 6,
        'permutations': 1
    }
    image_storage.new_group(subject_storage, cluster_assignments, **kwargs)
    # generate(images, subjects, cluster_assignments)

    import code
    import sqlite3
    conn = sqlite3.connect(image_storage.fname)
    print(conn.execute('select * from groups').fetchall())

    print('Press Enter to continue')
    input()

    image_storage = SQLImages(image_file)
    print(image_storage.list_groups())
    g = image_storage.get_group(0)
    g.generate_images(subject_storage, './images')
    code.interact(local={**globals(), **locals()})

if __name__ == '__main__':
    main()
