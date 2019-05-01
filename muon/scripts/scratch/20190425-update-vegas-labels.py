import click
import os
import csv
from tqdm import tqdm
from astropy.io import fits

from muon.config import Config
from muon.database.database import Database
from muon.subjects.storage import Storage


@click.command()
@click.argument('source_path')
@click.option('--config')
def main(source_path, config):
    print(1)
    if config:
        Config.new(config)
    database = Database()

    with database.conn as conn:
        query = "SELECT source_id FROM sources"
        cursor = conn.execute(query)
        sources = [row[0] for row in cursor]

        query = """
            INSERT INTO subject_labels (subject_id, label_name, label)
            VALUES (
                (SELECT subject_id FROM subjects WHERE source_id=? LIMIT 1),
                "vegas_cleaned",?);
        """

        # sources = ['SIMS150020.fits']
        print(sources)
        print(query)
        for source in sources:
            print(source)
            fname = os.path.join(source_path, source)
            data = []
            with fits.open(fname) as hdul:
                print('loading rows')
                for row in tqdm(hdul[1].data):
                    radius = row['MuRadius']
                    label = row['IsMuon']
                    id_ = (row['RunNum'], row['EventNum'], row['Telescop'])
                    id_ = 'run_{}_evt_{}_tel_{}'.format(*id_)
                    if source.startswith('SIM'):
                        id_ = 'sim_' + id_

                    if label == True:
                        # print(label, radius, id_)
                        if radius > 0.5:
                            data.append((id_, 1))
                        elif radius < 0.4:
                            data.append((id_, 0))
                    # print(row['IsMuon'], row['MuRadius'])
            print(len(data))

            print('Uploading data')
            print(query)
            print(data)
            conn.executemany(query, data)
            conn.commit()


if __name__ == '__main__':
    main()
