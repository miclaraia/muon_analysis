import click


from muon.subjects.storage import Storage
from muon.images.image_group import ImageGroup
from muon.database.database import Database


@click.group(invoke_without_command=True)
@click.argument('database')
def main(database):
    database = Database(database)


    subject_storage = Storage(database)

    kwargs = {
        'image_size': 36,
        'image_width': 6,
        'permutations': 1,
    }

    with database.conn as conn:
        group_id = database.ImageGroup.next_id(conn)
        subjects = database.Subject.list_subjects(conn)

    clusters = {0: subjects[:36*20]}
    group = ImageGroup.new(group_id, database, 'test', clusters, **kwargs)

    print(group)

    with database.conn as conn:
        import code
        code.interact(local={**globals(), **locals()})

main()
