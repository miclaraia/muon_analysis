import os
import logging
import pickle
from tqdm import tqdm
import click

from muon.database.database import Database

logger = logging.getLogger(__name__)


def insert_iter(agg_file, label_name):
    with open(agg_file, 'rb') as f:
        data = pickle.load(f)

    data = data['subjects']
    count = 0
    for s in tqdm(data):
        if len(data[s]) > 5:
            count += 1
            v = sum(data[s]) / len(data[s])
            yield s, labele_name, v
    print(count)


@click.group(invoke_without_command=True)
@click.argument('label-name')
@click.argument('agg-file')
def main(label_name, agg_file):
    database = Database()

    query = """
        INSERT INTO subject_labels (subject_id, label_name, label)
        VALUES (%s,%s,%s)
    """
    print(query)
    with database.conn as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, insert_iter(agg_file, label_name))

        conn.commit()


if __name__ == '__main__':
    main()
