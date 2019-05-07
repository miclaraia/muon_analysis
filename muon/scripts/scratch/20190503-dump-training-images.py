import click
import csv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from muon.database.database import Database
from muon.subjects.storage import Storage
from muon.subjects.subject import Subject


def get_subjects(conn, num, label, simulated):
    query = """
        CREATE TEMPORARY TABLE temp_muons (
            subject_id TEXT PRIMARY KEY,
            rand INTEGER
        )
    """
    conn.execute(query)

    query = """
        INSERT INTO temp_muons (subject_id, rand)
        SELECT S.subject_id, RANDOM()
        FROM subjects AS S
        INNER JOIN subject_labels AS L ON S.subject_id=L.subject_id
        INNER JOIN sources ON sources.source_id=S.source
        WHERE L.label_name="vegas_cleaned"
            AND L.label=?
            AND sources.source_type=?
        ;
    """
    simulated = {True: 1, False: 0}[simulated]
    print(query)
    conn.execute(query, (label, simulated))

    query = """
        DELETE FROM temp_muons
        WHERE subject_id IN (
            SELECT temp_muons.subject_id FROM temp_muons
                INNER JOIN image_subjects AS I
                    ON temp_muons.subject_id=I.subject_id
                INNER JOIN images ON images.image_id=I.image_id
                WHERE images.group_id=21 OR images.group_id=22
            );
    """
    print(query)
    conn.execute(query)

    # cursor = conn.execute('SELECT * from temp_muons LIMIT 50')
    # for row in cursor:
        # print(row)

    query = """
        SELECT S.subject_id, S.charge, L.label, S.source_id, S.source
        FROM temp_muons
        INNER JOIN subjects as S ON S.subject_id=temp_muons.subject_id
        INNER JOIN subject_labels AS L ON S.subject_id=L.subject_id
        WHERE L.label_name="vegas_cleaned"
        ORDER BY rand
        LIMIT ?
    """
    cursor = conn.execute(query, (num,))

    for row in cursor:
        fields = ['subject_id', 'charge', 'label', 'source_id', 'source_file']
        row = {fields[i]: row[i] for i in range(len(fields))}
        row['charge'] = np.fromstring(row['charge'], dtype=np.float32)

        subject = Subject(row['subject_id'], row['charge'])
        yield subject, row

    query = "DROP TABLE temp_muons"
    conn.execute(query)


def generate(subjects, path, manifest):
    os.mkdir(path)
    metadata = []
    for subject, meta in tqdm(subjects):
        fig = plt.figure(figsize=(2, 2))
        fig.tight_layout()
        ax = fig.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        subject.plot(ax)

        fname = os.path.join(path, '{}.jpg'.format(subject.id))
        fig.savefig(fname, dpi=150, quality=95)
        print(fname)

        del meta['charge']
        metadata.append(meta)

    with open(manifest, 'w') as f:
        fields = ['subject_id', 'label', 'source_id', 'source_file']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for meta in metadata:
            writer.writerow(meta)


@click.group(invoke_without_command=True)
@click.argument('save_dir')
def main(save_dir):
    n_muons = 5000
    n_nonmuons = 15000

    muon_dir = os.path.join(save_dir, 'muons')
    nonmuon_dir = os.path.join(save_dir, 'nonmuons')

    # if os.path.isdir(muon_dir):
        # print(muon_dir)
        # raise Exception('Muon path already exists')
    if os.path.isdir(nonmuon_dir):
        print(nonmuon_dir)
        raise Exception('Nonmuon path already exists')

    database = Database()

    with database.conn as conn:
        # subjects = get_subjects(conn, n_muons, 1, False)
        # manifest = os.path.join(save_dir, 'muon-manifest.csv')
        # generate(subjects, muon_dir, manifest)

        subjects = get_subjects(conn, n_nonmuons, 0, False)
        manifest = os.path.join(save_dir, 'nonmuon-manifest.csv')
        generate(subjects, nonmuon_dir, manifest)
            



if __name__ == '__main__':
    main()

