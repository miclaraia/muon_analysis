import pickle
import click

from redec_keras.models.multitask import MultitaskDEC
from redec_keras.models.redec import ReDEC
from redec_keras.models.decv2 import DECv2, Config
from muon.subjects.storage import Storage


@click.group(invoke_without_command=True)
@click.argument('model_save_dir')
@click.argument('subject_storage')
@click.argument('output')
def main(model_save_dir, subject_storage, output):
    """
    Generate a file containing cluster assignment information for each
    subject
    """
    if 'multitask' in model_save_dir:
        Model = MultitaskDEC
    elif 'redec' in model_save_dir:
        Model = ReDEC
    elif 'supervised' not in model_save_dir:
        Model = DECv2

    config = Config.load(model_save_dir+'/config.json')
    with open(config.splits_file, 'rb') as file:
        splits = pickle.load(file)
    dec = Model.load(model_save_dir, splits['train'][0])

    subject_storage = Storage(subject_storage)
    subjects = subject_storage.get_all_subjects()
    subject_ids = subjects.keys()
    x = subjects.get_x(subject_ids)

    cluster_pred = zip(subject_ids, dec.predict_clusters(x))
    cluster_assignments = {}
    for subject_id, cluster in cluster_pred:
        if cluster not in cluster_assignments:
            cluster_assignments[cluster] = []
        cluster_assignments[cluster].append(subject_id)

    import code
    code.interact(local={**globals(), **locals()})
    with open(output, 'wb') as f:
        pickle.dump(cluster_assignments, f)

    

