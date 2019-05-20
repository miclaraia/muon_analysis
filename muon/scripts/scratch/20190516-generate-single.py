import click
from tqdm import tqdm
import random
import os

from muon.subjects.storage import Storage
from muon.database.database import Database
from muon.images.image_group import ImageGroup
from muon.images.image_group_single import SingleImageGroup


"""
    Need subjects for the single image classification
    by experts. The images need to be from the test set,
    as in the classifier should not have been trained on
    any of these images.
    5,000 should be real data, and 5,000 should be simulated

    In each of these sets, this will sample a specific subset based on
    existing labels.
    1/8: cleaned muon
    1/8: cleaned nonmuon
    remaining: original labels, nonmuon

    Randomly sampling did not put enough muons in the data set from the cleaned
    labels (7/10000).
"""


def get_subjects(conn, num, groups):
    query = """
        CREATE TEMPORARY TABLE temp_muons (
            subject_id UUID PRIMARY KEY,
            rand INTEGER
        )
    """
    with conn.cursor() as cursor:
        cursor.execute(query)

    query = """
        INSERT INTO temp_muons (subject_id, rand)
        SELECT S.subject_id, RANDOM()
        FROM subjects AS S
        INNER JOIN image_subjects AS I_S ON S.subject_id=I_S.subject_id
        INNER JOIN images AS I ON I_S.image_id=I.image_id
        WHERE
            S.split_id=1
            AND I.group_id IN ({})
        GROUP BY S.subject_id
        ;
    """.format(','.join(['%s' for _ in groups]))

    print(query)
    with conn.cursor() as cursor:
        cursor.execute(query, tuple(groups))

    # cursor = conn.execute('SELECT * from temp_muons LIMIT 50')
    # for row in cursor:
        # print(row)

    count = 0
    query = """
        SELECT T.subject_id FROM temp_muons as T
        INNER JOIN subject_labels AS L ON T.subject_id=L.subject_id
        WHERE L.label_name=%s and L.label=%s
        ORDER BY T.rand LIMIT %s
    """
    print(query)
    with conn.cursor() as cursor:
        cursor.execute(query, ('vegas_cleaned', 1, num//8,))

        for row in cursor:
            count += 1
            yield row[0]
    print(count)
    print(query)

    with conn.cursor() as cursor:
        cursor.execute(query, ('vegas_cleaned', 0, num//8,))

        for row in cursor:
            count += 1
            yield row[0]
    print(count)
    print(query)

    with conn.cursor() as cursor:
        cursor.execute(query, ('vegas2', 0, num - count,))

        for row in cursor:
            count += 1
            if count > num:
                break
            yield row[0]
    print(count)

    query = "DROP TABLE temp_muons"
    with conn.cursor() as cursor:
        cursor.execute(query)


@click.group()
def cli():
    pass


@cli.command()
def init_groups():
    N = 5000
    group_splits = {'sims': [13], 'real': [10, 11, 12]}
    descriptions = {'sims': 'Single images (sims)',
                    'real': 'Single images (real)'}

    groups = []

    database = Database()
    with database.conn as conn:
        for k in ['sims', 'real']:
            subject_ids = []
            for s in tqdm(get_subjects(conn, N, group_splits[k])):
                subject_ids.append(s)
            description = descriptions[k]

            group = SingleImageGroup.new(
                database, subject_ids, description=description)
            groups.append(group)

    print(groups)


@cli.command()
@click.argument('groups')
def generate(groups):
    print('Use pipeline/workers.py instead!')


if __name__ == '__main__':
    cli()
