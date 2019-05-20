import psycopg2
import sqlite3
import click
from tqdm import tqdm


@click.group(invoke_without_command=True)
@click.argument('sqlite_file')
def main(sqlite_file):
    sql_conn = sqlite3.connect(sqlite_file)
    pg_conn = psycopg2.connect(
        host='localhost',
        database='muon_data')

    fields = {
        'images': [
            'fig_dpi', 'fig_offset', 'fig_height', 'fig_width',
            'fig_rows', 'fig_cols'],
    }

    transfer(fields['images'], sql_conn, pg_conn)
    pg_conn.commit()



def transfer(fields, sql_conn, pg_conn):

    query_get = 'SELECT image_id,{} FROM images'.format(','.join(fields))
    query_set = 'UPDATE images SET {} WHERE image_id=%s'

    query_set = query_set.format(','.join(['{}=%s'.format(f) for f in fields]))

    print(query_get)
    sql_cur = sql_conn.execute(query_get)
    def data_iter():
        i = 0
        for row in tqdm(sql_cur):
            row = (*row[1:], row[0])
            if i < 10:
                print(row)
                i += 1
            yield row

    print(query_set)
    with pg_conn.cursor() as pg_cur:
        pg_cur.executemany(query_set, data_iter())


if __name__ == '__main__':
    main()
