import click

from muon.database.database import Database


def main():
    database = Database()
    with database.conn as conn:
        query = """
            ALTER TABLE sources ADD COLUMN source_type
                INTEGER NOT NULL DEFAULT 0;
        """
        conn.execute(query)
        conn.commit()

        query = "SELECT source_id,source_type FROM sources"
        cursor = conn.execute(query)
        query = "UPDATE sources SET source_type=? WHERE source_id=?"

        for source_id, source_type in cursor:
            if 'SIMS' in source_id and source_type != 1:
                conn.execute(query, (1, source_id))
        conn.commit()


if __name__ == '__main__':
    main()
