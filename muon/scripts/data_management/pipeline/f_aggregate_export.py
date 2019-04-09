import click
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.debug(1)
logger.info(2)
logger.warn(3)

from muon.database.database import Database
from muon.project.parse_export import Aggregate


class config:
    tool_name = 'Tool name'
    task_A = 'T0'
    task_B = ['T1', 'T2']
    launch_date = '2019-03-10'

    time_format = '%Y-%m-%d %H:%M:%S %Z'
    image_groups = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    image_dir = 'path/to/images'

    task_map = {
        'all_muons': ['All are Muons'],
        'most_muons': ['Majority are Muons', 'No clear majority'],
        'most_nonmuons': ['Majority are **not** Muons'],
        'no_muons': ['**None** are Muons']
    }


@click.group(invoke_without_command=True)
@click.argument('database_file')
@click.argument('export_file')
@click.argument('out_file')
def main(database_file, export_file, out_file):
    database = Database(database_file)
    agg = Aggregate(config, database)
    agg.aggregate(export_file)

    data = {'subjects': agg.subjects, 'images': agg.images}

    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


main()
