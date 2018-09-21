#!/usr/bin/env python

import click
import code
import json

import numpy as np

from zootools.data_parsing import Parser, Reducer


@click.group(invoke_without_command=True)
@click.argument('fname')
@click.option('--output')
@click.option('--counts', is_flag=True)
def main(fname, output, **kwargs):
    #fname = '/home/michael/Downloads/muon-hunter-classifications-2.csv'
    task_map = {
        'yes': ['Yes', 'Yes!', 'yes'],
        'no': ['No', 'No.']
    }
    annotations = Parser(['T1'], task_map, '2010-01-01')(fname)
    tasks, subjects, values, counts = Reducer.count(annotations)

    majority = {task: np.argmax(counts[task], axis=1) for task in counts}
    fraction = {task: counts[task]/np.sum(counts[task], axis=1)[:,None] for
            task in counts}

    if output:
        obj = {
            'majority': majority,
            'fraction': fraction,
            'tasks': tasks,
            'subjects': subjects,
            'values': values}
        if kwargs['counts']:
            obj.update({'counts': counts})
        with open(output, 'w') as file:
            json.dump(obj, file)

    code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
