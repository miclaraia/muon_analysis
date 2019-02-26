import csv
import json
import click
from tqdm import tqdm

from muon.subjects.database import Database


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('subject_export')
def main(database_file, subject_export):
    with open(subject_export, 'r') as file:
        existing_subjects = {}
        for row in csv.DictReader(file):
            image_id = json.loads(row['metadata'])['id']
            zoo_id = int(row['subject_id'])

            if image_id in existing_subjects:
                print('image {} zooid {}'.format(image_id, zoo_id))
                raise Exception('Found duplicate image-subject mapping')
            else:
                existing_subjects[image_id] = zoo_id

    database = Database(database_file)

    wrong_id = 0
    no_zoo = 0
    with database.conn as conn:
        for image in tqdm(database.Image.get_group_images(conn, 10)):
            image_id = image.image_id

            if image_id in existing_subjects:
                # this image is in the zooniverse
                zoo_id = existing_subjects[image_id]
                if image.zoo_id != zoo_id:
                    # this image does not have the right zoo_id
                    wrong_id += 1
                    print('wrong zoo id {}'.format(image_id))
                    image.zoo_id = zoo_id
                    database.update_zooid(conn, image)

            else:
                if image.zoo_id is not None:
                    # This image is not on the zooniverse export
                    # but it does have an image_id
                    # probably tried to upload it and failed
                    no_zoo += 1
                    print('no zooniverse subject {} {}'.format(image_id, zoo_id))
                    image.zoo_id = None
                    database.Image.update_zooid(conn, image)

        conn.commit()

    print('wrong id {} no zoo subject {}'.format(wrong_id, no_zoo))


main()
