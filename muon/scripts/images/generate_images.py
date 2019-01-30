
import click
import logging
import pickle

from muon.project.hdf_images import HDFImages
from muon.subjects.storage import Storage

logger = logging.getLogger(__name__)


def load_cluster_assignments(cluster_assignment_file):
    with open(cluster_assignment_file, 'rb') as f:
        return pickle.load(f)


@click.group(invoke_without_command=True)
def main(image_file, subject_file, cluster_assignment_file):
    hdf_images = HDFImages(image_file)
    subject_storage = Storage(subject_file)
    cluster_assignments = load_cluster_assignments(cluster_assignment_file)

    kwargs = {
        'image_size': 9,
        'width': 3,
        'permutations': 1
    }
    hdf_images.new_group(subject_storage, cluster_assignments, **kwargs)
    # generate(images, subjects, cluster_assignments)

