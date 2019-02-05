import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.subjects import Subject
from muon.subjects.parsing import ParseFits


@click.group(invoke_without_command=True)
@click.argument('input_file')
@click.argument('output_file')
def main(input_file, output_file):
    #for run, event, charge in HDFParseQi.raw_file(input_file):
    storage = Storage(output_file)

    parser = ParseFits.parse_file(input_file)
    skipped = storage.add_subjects(parser)
    print('skipped subjects:')
    print(skipped)

if __name__ == '__main__':
    main()
