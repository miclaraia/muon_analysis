import click
import os
from tqdm import tqdm
from astropy.io import fits

from muon.config import Config
from muon.database.database import Database


def get_existing_labels(conn, source):
    query = """
        SELECT S.source_id, L.label
        FROM subjects as S
        INNER JOIN subject_labels as L
            INDEXED BY subject_label
            ON L.subject_id=S.subject_id
        WHERE S.source=?
            AND L.label_name="vegas2"
    """

    print(query)
    cursor = conn.execute(query, (source,))
    data = {}
    for row in tqdm(cursor):
        data[row[0]] = row[1]
    return data


def insert_new_labels(conn, data):
    query = """
        INSERT INTO subject_labels (subject_id, label_name, label)
        VALUES (
            (SELECT subject_id FROM subjects WHERE source_id=? LIMIT 1),
            "vegas2",?);
    """

    def data_iter():
        for row in tqdm(data):
            yield row

    with conn.cursor() as cursor:
        cursor.executemany(query, data_iter())


def update_label(conn, source_id, label):
    query = """
        UPDATE subject_labels as L
        INNER JOIN subjects as S
            ON S.subject_id=L.subject_id
        SET L.label=?
        WHERE S.source_id=? and L.label_name=?
    """

    def data_iter():
        for label, source_id in tqdm(data):
            yield label, source_id, 'vegas2'

    with conn.cursor() as cursor:
        cursor.executemany(query, data_iter())




@click.command()
@click.argument('source_path')
@click.option('--config')
def main(source_path, config):
    print(1)
    if config:
        Config.new(config)
    database = Database()

    skip = [
        '84403.fits',
        '84404.fits',
        '84625.fits',
        '85496.fits',
        '85497.fits',
        '88784.fits',
        '88785.fits',
        '89245.fits',
        '89734.fits',
        '89735.fits',
        '90074.fits',
        '90075.fits',
        '91965.fits',
        '91967.fits',
        '91970.fits',
        '91977.fits',
        '91994.fits',
        '91995.fits',
        '92002.fits',
        '92152.fits',
        '92156.fits',
        '92175.fits',
        '92185.fits',
        '92195.fits',
        ]

    with database.conn as conn:
        query = "SELECT source_id FROM sources"
        print(query)
        cursor = conn.execute(query)
        sources = []
        # for row in cursor:
            # if row[0] == 'SIMS140041.fits':
                # sources.append(row[0])
                # break
        for row in cursor:
            sources.append(row[0])

        # sources = ['SIMS150020.fits']
        print(sources)
        for source in sources:
            if source in skip:
                continue

            print(source)
            insert = []
            update = []
            print('Lodaing existing labels')
            existing_labels = get_existing_labels(conn, source)

            print('Existing labels:', len(existing_labels))
            fname = os.path.join(source_path, source)
            with fits.open(fname) as hdul:
                print('loading rows')
                for row in tqdm(hdul[1].data):
                    label = int(row['IsMuon'])
                    id_ = (row['RunNum'], row['EventNum'], row['Telescop'])
                    id_ = 'run_{}_evt_{}_tel_{}'.format(*id_)
                    if source.startswith('SIM'):
                        id_ = 'sim_' + id_

                    cond = id_ not in existing_labels or \
                        existing_labels[id_] != label
                    if id_ not in existing_labels:
                        insert.append((id_, label))
                    elif existing_labels[id_] != label:
                        update.append((id_, label))

            def tqdm_iter(data):
                for item in tqdm():
                    yield item

            print('Inserting')
            for id_, label in tqdm(insert):
                insert_new_label(conn, id_, label)
            print('Updating')
            for id_, label in tqdm(update):
                update_label(conn, id_, label)
            conn.commit()


if __name__ == '__main__':
    main()
