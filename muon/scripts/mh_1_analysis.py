#!/usr/bin/env python

import click
import code
import json
import csv
import os

from tqdm import tqdm
import numpy as np

from zootools.data_parsing import Parser, Reducer
from zootools.data_output import HDFOutput


@click.group()
def cli():
    pass


@cli.command()
@click.argument('directory')
@click.argument('subjects_csv')
def check_image_fname(directory, subjects_csv):
    with open(subjects_csv, 'r') as file:
        reader = csv.DictReader(file)
        subjects = {}
        for row in tqdm(reader):
            meta = json.loads(row['metadata'])
            fname = None
            for value in meta.values():
                if type(value) is str and 'jpeg' in value:
                    fname = value
                    break
            subjects[fname] = row['subject_id']
            
            
    fname_data = []
    for dir, _, filenames in os.walk(directory):
        if dir != '.':
            for filename in filenames:
                if os.path.splitext(filename)[1] == '.jpeg':
                    fname_data.append(filename)
    fname_data = set(fname_data)
    subject_data = set(subjects.keys())

    print(len(fname_data))
    print(len(fname_data & subject_data))
    print(len(fname_data - subject_data))
    print(len(fname_data ^ subject_data))

    



def parse_subject_metadata(fname):
    mapping = {
        'non_muon': 0,
        'muon': 1
    }
    with open(fname, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subject = row['subject_id']
            image_file = json.loads(row['metadata'])['Filename']


@cli.command()
@click.argument('fname')
@click.argument('output')
def subject_metadata(fname, output):
    from muon.swap.muon_metadata import SubjectID
    data = SubjectID.collect_data(fname)

    with open(output, 'w') as file:
        fieldnames = ['subject', 'run', 'evt', 'tel']
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        for subject, item in data.items():
            writer.writerow({'subject': subject, **item})

    code.interact(local={**globals(), **locals()})
    

            


#@click.group(invoke_without_command=True)
@cli.command()
@click.argument('fname')
@click.argument('output')
@click.option('--task', default='T1')
#@click.option('--counts', is_flag=True)
def main(fname, output, task, **kwargs):
    #fname = '/home/michael/Downloads/muon-hunter-classifications-2.csv'

    task_map = {
        'yes': ['Yes', 'Yes!', 'yes'],
        'no': ['No', 'No.']
    }
    annotations = Parser(task, task_map, '2010-01-01')(fname)
    tasks, subjects, values, counts = Reducer.count(annotations)

    values = values[task]
    counts = counts[task]
    majority = np.argmax(counts, axis=1)
    fraction = counts/np.sum(counts, axis=1)[:,None]

    HDFOutput(task, values, subjects, output).\
            create(None, counts, majority, fraction)
#     task_map = {
#         'yes': ['Yes', 'Yes!', 'yes'],
#         'no': ['No', 'No.']
#     }
#     annotations = Parser(['T1'], task_map, '2010-01-01')(fname)
#     tasks, subjects, values, counts = Reducer.count(annotations)
# 
#     majority = {task: np.argmax(counts[task], axis=1) for task in counts}
#     fraction = {task: counts[task]/np.sum(counts[task], axis=1)[:,None] for
#             task in counts}
# 
#     if output:
#         obj = {
#             'majority': {task: majority[task].tolist() for task in majority},
#             'fraction': {task: fraction[task].tolist() for task in fraction},
#             'tasks': tasks,
#             'subjects': subjects,
#             'values': values}
#         if kwargs['counts']:
#             obj.update({'counts': counts})
# 
#         with open(output, 'w') as file:
#             print('Saving to %s' % output)
#             json.dump(obj, file)
# 
#     code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    cli()
