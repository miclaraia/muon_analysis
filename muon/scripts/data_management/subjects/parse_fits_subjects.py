import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.subjects import Subject
from muon.subjects.parsing import ParseFits
from muon.database.database import Database


@click.group(invoke_without_command=True)
@click.argument('input_file')
@click.argument('database_file')
def main(input_file, database_file):
    #for run, event, charge in HDFParseQi.raw_file(input_file):
    database = Database(database_file)
    storage = Storage(database)

    parser = ParseFits.parse_file(input_file)
    storage.add_subjects(parser)


if __name__ == '__main__':
    main()
