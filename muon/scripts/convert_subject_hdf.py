import click
import h5py
from muon.utils.subjects import Subject_Data
from muon.subjects.storage import Storage
from muon.subjects import Subject


@click.group(invoke_without_command=True)
@click.argument('in_file')
@click.argument('out_file')
def main(in_file, out_file):
    sd = Subject_Data(in_file)

    def subjects():
        for subject, (run, event, tel), charge in sd:
            metadata = {
                'run': int(run),
                'evt': int(event),
                'tel': int(tel),
            }
            subject = Subject(subject, charge, metadata)
            yield subject

    storage = Storage(out_file)
    skipped = storage.add_subjects(subjects())

    print('skipped these subjects:', skipped)


if __name__ == '__main__':
    main()
