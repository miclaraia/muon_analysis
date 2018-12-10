import click
import csv
from tqdm import tqdm
import logging
from tqdm import tqdm

from muon.project.hdf_images import HDFImages
from zootools.data_parsing import Parser, GridAggregate

from swap.utils.config import Config
from swap.utils.control import SWAP

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers[0].setLevel(logging.INFO)


@click.group(invoke_without_command=True)
@click.option('--export', required=True)
@click.option('--images', required=True)
@click.option('--score_output')
@click.option('--label_output')
@click.option('--threshold', type=float, default=0.5)
@click.option('--golds', required=True)
def main(export, images, score_output, label_output, golds, threshold):
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
    aggregated = grid_aggregate.aggregate_swap(parser(export))


    # Load gold labels from csv
    with open(golds, 'r') as f:
        reader = csv.DictReader(f)
        golds = []
        for row in reader:
            # need to cast subjects and labels as ints for swap to work
            golds.append((int(row['subject']), int(row['label'])))

    config = Config(name='muon')
    swap = SWAP(config)
    swap.apply_golds(golds)

    i = 0
    for subject, user, cl in tqdm(aggregated):
        swap.classify(user, subject, cl, i)
        i += 1
    swap()

    if score_output:
        with open(score_output, 'w') as f:
            writer = csv.DictWriter(f, ['subject', 'label'])
            writer.writeheader()
            for subject in swap.subjects.iter():
                writer.writerow({'subject': subject.id, 'label': subject.score})

    if label_output:
        with open(label_output, 'w') as f:
            writer = csv.DictWriter(f, ['subject', 'label'])
            writer.writeheader()
            for subject in swap.subjects.iter():
                writer.writerow({
                    'subject': subject.id,
                    'label': int(subject.score>threshold)})


if __name__ == '__main__':
    main()

