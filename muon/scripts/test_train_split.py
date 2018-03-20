#!/usr/bin/env python

import os
import csv
import random

from muon.project.images import Images, MultiGroupImages
import muon.project.panoptes as pan

import click


@click.group(invoke_without_command=True)
@click.argument('groups')
@click.argument('output-folder')
def main(groups, output_folder):
    print('groups', groups)
    groups = [int(i) for i in groups.split(',')]
    images = MultiGroupImages(groups)
    print('images', images)

    subjects = []
    for image in images.iter():
        subjects.extend(image.subjects)

    subjects = sorted(list(set(subjects)))
    print('collected %d subjects' % len(subjects))


    random.shuffle(subjects)

    test_size = round(.25*len(subjects))
    test_set = subjects[:test_size]
    train_set = subjects[test_size:]

    make_csv(test_set, 'test', output_folder)
    make_csv(train_set, 'train', output_folder)


def make_csv(subjects, name, output):
    with open(os.path.join(output, '%s.csv' % name), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['subject'])
        for s in subjects:
            writer.writerow([s])


if __name__ == '__main__':
    main()
