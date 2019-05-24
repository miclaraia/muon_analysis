import os
import logging
import pickle
from tqdm import tqdm

from muon.database.database import Database

logger = logging.getLogger(__name__)


root = os.getenv('MUOND')
export_file = os.path.join(root, 'zooniverse/MH2/20190520/agg.pkl')


def insert_iter():
    with open(export_file, 'rb') as f:
        data = pickle.load(f)

    data = data['subjects']
    count = 0
    for s in tqdm(data):
        if len(data[s]) > 5:
            count += 1
            v = sum(data[s]) / len(data[s])
            yield s, 'volunteer_majority_20190520', v
    print(count)


def main():
    database = Database()

    query = """
        INSERT INTO subject_labels (subject_id, label_name, label)
        VALUES (%s,%s,%s)
    """
    print(query)
    with database.conn as conn:
        with conn.cursor() as cursor:
            cursor.executemany(query, insert_iter())

        conn.commit()


if __name__ == '__main__':
    main()
