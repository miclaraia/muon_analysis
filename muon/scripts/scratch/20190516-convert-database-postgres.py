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
            'image_id', 'group_id', 'cluster', 'metadata',
            'zoo_id', 'fig_dpi', 'fig_offset', 'fig_height',
            'fig_rows', 'fig_cols'],
        'image_subjects': [
            'subject_id', 'image_id', 'group_id', 'image_index'],
        'image_groups': [
            'group_id', 'group_type', 'cluster_name', 'image_count',
            'image_size', 'image_width', 'description', 'permutations'],
        'subjects': [
            'subject_id', 'source_id', 'source', 'charge', 'batch_id',
            'split_id'],
        'subject_clusters': [
            'subject_id', 'cluster_name', 'cluster'],
        'subject_labels': [
            'subject_id', 'label_name', 'label'],
        'sources': [
            'source_id', 'source_type', 'hash', 'updated'],
        'workers': [
            'job_id', 'job_type', 'job_status'],
        'worker_images': [
            'job_id', 'image_id'],
    }

    for table in sorted(fields):
        transfer_general(table, sql_conn, pg_conn, fields[table])
        pg_conn.commit()


def transfer_subjects(sql_conn, pg_conn):
    query_get = """
        SELECT subject_id, source_id, source, charge, batch_id, split_id
        FROM subjects
    """
    query_set = """
        INSERT INTO subjects
        (subject_id, source_id, source, charge, batch_id, split_id)
        VALUES (%s,%s,%s,%s,%s,%s)
    """

    sql_cur = sql_conn.execute(query_get)
    with pg_conn.cursor() as pg_cur:
        pg_cur.executemany(query_set, sql_cur)


def transfer_images(sql_conn, pg_conn):
    query_get = """
        SELECT
            image_id, group_id, cluster, metadata, zoo_id
            fig_dpi, fig_offset, fig_height, fig_width, fig_rows,
            fig_cols
        FROM  imagese
    """
    query_set = """
        INSERT INTO images
            (image_id, group_id, cluster, metadata, zoo_id
            fig_dpi, fig_offset, fig_height, fig_width, fig_rows,
            fig_cols)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    sql_cur = sql_conn.execute(query_get)
    with pg_conn.cursor() as pg_cur:
        pg_cur.executemany(query_set, sql_cur)


def transfer_general(table, sql_conn, pg_conn, fields):
    print(table, fields)

    query_get = 'SELECT {} FROM {}'.format(','.join(fields), table)
    print(query_get)
    query_set = 'INSERT INTO {} ({}) VALUES ({})'.format(
        table, ','.join(fields), ','.join(['%s' for _ in fields]))
    print(query_set)

    sql_cur = sql_conn.execute(query_get)

    def data_iter():
        for row in tqdm(sql_cur):
            yield row

    with pg_conn.cursor() as pg_cur:
        pg_cur.executemany(query_set, data_iter())


def transfer_image_groups(sql_conn, pg_conn):
    query_get = """
        SELECT subject_id, image_id, group_id, image_index
        FROM image_subjects
    """
    query_set = """
        INSERT INTO image_subjects
            (subject_id, image_id, group_id, image_index)
        VALUES (%s,%s,%s,%s)
    """
    sql_cur = sql_conn.execute(query_get)
    with pg_conn.cursor() as pg_cur:
        pg_cur.executemany(query_set, sql_cur)


if __name__ == '__main__':
    main()
