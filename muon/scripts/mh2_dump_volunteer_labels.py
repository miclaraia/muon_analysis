
import click
import csv
from tqdm import tqdm

from muon.project.hdf_images import HDFImages
from zootools.data_parsing import Parser, GridAggregate


@click.group(invoke_without_command=True)
@click.argument('export')
@click.argument('images')
@click.argument('output')
def main(export, images, output):
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

    images = HDFImages(images)
    subject_map = {}
    image_group_map = {}
    for group in images.list_groups():
        image_group = images.get_group(group)
        for image in image_group.iter():
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

    grid_aggregate = GridAggregate(
        subject_map, location_callback, answer_map, 'T0', ('T1', 'T2'))

    print(sum([len(v) for v in subject_map.values()]), len(image_group_map))
    input()
    aggregated = grid_aggregate(parser(export))
    print(len(aggregated))

    with open(output, 'w') as file:
        writer = csv.DictWriter(file, ['subject', 'label'])
        writer.writeheader()
        for s in tqdm(aggregated):
            annotations = aggregated[s]
            label = sum(annotations)/len(annotations)
            writer.writerow({'subject': s, 'label': int(label>.5)})


if __name__ == '__main__':
    main()
