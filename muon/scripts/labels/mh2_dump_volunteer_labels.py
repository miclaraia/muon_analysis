
import click
import csv
from tqdm import tqdm
import logging
import time

from muon.project.hdf_images import HDFImages
from zootools.data_parsing import Parser, GridAggregate

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers[0].setLevel(logging.INFO)


@click.group(invoke_without_command=True)
@click.option('--export', required=True)
@click.option('--images', required=True)
@click.option('--majority_output')
@click.option('--fraction_output')
@click.option('--first_output')
@click.option('--threshold', type=float, default=0.5)
def main(export, images, majority_output, fraction_output, first_output, threshold):
    tasks = {'T0': 'choice',
             'T1': 'grid',
             'T2': 'grid'}
    task_map = {
        'all_muons': ['All Muons'],
        'most_muons': ['Majority are Muons', 'No clear majority'],
        'most_nonmuons': ['Minority are Muons'],
        'no_muons': ['No Muons']
    }
    launch_date = '2018-02-15'

    parser = Parser(tasks, task_map, launch_date)

    logger.info('loading images')
    images = HDFImages(images)
    subject_map = {}
    image_group_map = {}
    for group in images.list_groups():
        image_group = images.get_group(group)
        for image in tqdm(image_group.iter()):
            i = image.zoo_id
            subject_map[i] = list(image.subjects)
            image_group_map[i] = group

    answer_map = {'all_muons': 'all_true',
                  'most_muons': 'true',
                  'most_nonmuons': 'false',
                  'no_muons': 'all_false'}

    def location_callback(zoo_subject_id, x, y):
        image_group = images.get_group(image_group_map[zoo_subject_id])
        return image_group.get_zoo(zoo_subject_id).at_location(x, y)

    logger.info('initializing GridAggregate')
    grid_aggregate = GridAggregate(
        subject_map, location_callback, answer_map, 'T0', ('T1', 'T2'))

    print(sum([len(v) for v in subject_map.values()]), len(image_group_map))
    print('Waiting 10 seconds')
    time.sleep(10)
    aggregated = grid_aggregate.aggregate(parser(export))
    print(len(aggregated))

    if majority_output:
        logger.info('Writing majority output')
        with open(majority_output, 'w') as file:
            writer = csv.DictWriter(file, ['subject', 'label'])
            writer.writeheader()
            for s in tqdm(aggregated):
                annotations = aggregated[s]
                label = sum(annotations)/len(annotations)
                writer.writerow({'subject': s, 'label': int(label>threshold)})

    if fraction_output:
        logger.info('Writing fraction output')
        with open(fraction_output, 'w') as file:
            writer = csv.DictWriter(file, ['subject', 'label'])
            writer.writeheader()
            for s in tqdm(aggregated):
                annotations = aggregated[s]
                fraction = sum(annotations)/len(annotations)
                writer.writerow({'subject': s, 'label': fraction})


    if first_output:
        logger.info('Writing first output')
        with open(first_output, 'w') as file:
            writer = csv.DictWriter(file, ['subject', 'label'])
            writer.writeheader()
            for s in tqdm(aggregated):
                writer.writerow({'subject': s, 'label': aggregated[s][0]})


if __name__ == '__main__':
    main()
