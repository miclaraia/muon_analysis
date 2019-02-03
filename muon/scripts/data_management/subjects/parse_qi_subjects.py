
import click
from tqdm import tqdm

from muon.subjects.storage import Storage
from muon.subjects.subjects import Subject
from muon.subjects.parsing import HDFParseQi


def subject_generator(parser):
    for (run, event, tel), charge in parser:
        metadata = {
            'run': int(run),
            'evt': int(event),
            'tel': int(tel)
        }
        yield Subject(None, charge, metadata)


@click.group(invoke_without_command=True)
@click.argument('input_file')
@click.argument('output_file')
def main(input_file, output_file):
    #for run, event, charge in HDFParseQi.raw_file(input_file):
    storage = Storage(output_file)

    parser = HDFParseQi.raw_file(input_file)
    skipped = storage.add_subjects(subject_generator(parser))
    print('skipped subjects:')
    print(skipped)


    s2 = Storage('100-subjects.h5')
    sample = storage.get_all_subjects()._sample_s(100)
    s2.add_subjects(sample.list())

    import code
    code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()

