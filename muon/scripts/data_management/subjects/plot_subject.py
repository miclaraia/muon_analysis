from muon.subjects.storage import Storage
from muon.subjects.database import Database

import click
import matplotlib.pyplot as plt

@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('subject_id')
def main(database_file, subject_id):
    database = Database(database_file)
    storage = Storage(database)
    subjects = storage.get_subjects([subject_id])

    fig = subjects.plot_subjects(w=1)
    fig.savefig('subject_{}.png'.format(subject_id))

main()
