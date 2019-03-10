import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.subjects import Subject
from muon.subjects.parsing import ParseFits
from muon.database.database import Database


@click.group(invoke_without_command=True)
@click.argument('input_file')
@click.argument('database_file')
@click.option('--batch', nargs=1, type=int)
def main(input_file, database_file, batch):
    database = Database(database_file)
    storage = Storage(database)

    parser = ParseFits.parse_file(input_file)

    subjects = []
    def wrapper():
        for subject in parser:
            subjects.append(subject)
            yield subject

    if batch:
        storage.add_subjects(wrapper(), batch)
    else:
        storage.add_batch(wrapper())

    subject_labels = [(subject.id, subject.y) for subject in subjects]
    storage.add_labels('vegas', subject_labels)


if __name__ == '__main__':
    main()
