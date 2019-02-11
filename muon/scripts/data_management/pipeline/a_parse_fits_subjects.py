import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.subjects import Subject
from muon.subjects.parsing import ParseFits
from muon.subjects.database import Database


@click.group(invoke_without_command=True)
@click.argument('input_file')
@click.argument('database_file')
def main(input_file, database_file):
    database = Database(database_file)
    storage = Storage(database)

    parser = ParseFits.parse_file(input_file)

    subjects = []
    def wrapper():
        for subject in parser:
            subjects.append(subject)
            yield subject

    storage.add_subjects(wrapper())

    labels = [(subject.id, subject.y) for subject in subjects]
    storage.add_labels('vegas', labels)


if __name__ == '__main__':
    main()
