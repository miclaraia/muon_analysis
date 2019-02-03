
import click
import random
import pickle

from muon.subjects.storage import Storage


@click.group(invoke_without_command=True)
@click.argument('storage_file')
@click.argument('cluster_out')
def main(storage_file, cluster_out):

    storage = Storage(storage_file)
    subjects = storage.get_all_subjects()

    keys = subjects.keys()

    width = 3
    image_size = 9

    cluster_assignment = {i: [] for i in range(10)}
    for k in keys:
        cluster_assignment[random.randrange(0, 10)].append(k)

    with open(cluster_out, 'wb') as file:
        pickle.dump(cluster_assignment, file)

if __name__ == '__main__':
    main()
