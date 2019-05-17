from muon.database.database import Database
from muon.images.image_group import ImageGroup
from muon.config import Config

import click
import os
from tqdm import tqdm

#fname = '/home/michael/Documents/Zooniverse/repos/muon/data/subjects/ralph_subjects/data.db'
#fname = os.path.join(os.getenv('MUON'), 'subjects-main/data-main.db')
config = os.path.join(os.getenv('MUON'), 'scripts/scratch/default_config.yml')
config = Config.new(config)
database = Database()
#database = Database(os.path.join(os.getenv('MUOND'), 'subjects-main/data-main.db'))


ImageGroup.load(21, database).images.load_all()
group = ImageGroup.load(21, database)

for image in tqdm(group.images.gen_iter('.')):
    pass
# with database.conn as conn:
    # cursor = database.Image.get_group_images(conn, 11)
    # for i in tqdm(cursor):
        # pass

    # images = []
    # for i, image in enumerate(cursor):
        # images.append(image)

        # if i>50:
            # break
    # print(images)



import code
code.interact(local={**globals(), **locals()})


