#!/usr/bin/env python

from swap.db import DB
import swap.config
from muon.ui import ui

import argparse
import csv
import json
import re
import os

import logging
logger = logging.getLogger(__name__)

parse = re.compile('run([0-9]+)_evt([0-9]+).jpeg')


def get_path():
    p = os.path
    path = p.dirname(p.abspath(__name__))
    return path


def swap_config(config):
    swap.config.import_config(config)
    swap.config.logger.init()


def upload_data(data):
    db = DB()
    requests = []

    def write():
        nonlocal requests
        print('writing')
        if len(requests) > 0:
            db.subjects.bulk_write(requests)
            requests = []

    for subject, metadata in data.items():
        r = db.subjects.update_metadata(subject, metadata, False)
        requests.append(r)

        if len(requests) > 1e4:
            write()
    write()


def get_metadata(fname):
    data = {}

    with open(fname, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subject, metadata = parse_row(row)
            data[subject] = metadata

    return data


def parse_row(row):
    meta = json.loads(row['metadata'])

    subject = int(row['subject_id'])
    fname = meta['Filename']

    print('subject %d fname %s' % (subject, fname))

    run_, evt = parse_fname(fname)

    metadata = {
        'run': run_,
        'event': evt,
    }

    print(metadata)

    return subject, metadata


def parse_fname(fname):
    run_, evt = parse.search(fname).groups()
    run_ = int(run_)
    evt = int(evt)
    return run_, evt


def main():
    parser = argparse.ArgumentParser()
    interface = Interface()

    interface.options(parser)
    args = parser.parse_args()
    interface.call(args)


if __name__ == '__main__':
    main()

    #######################################################################
    #####   Interface   ###################################################
    #######################################################################

class Interface(ui.Interface):
    command = 'metadata'

    def options(self, parser):
        parser.add_argument('config', help='Muon Hunters swap config file')
        parser.add_argument('file', help='subject csv dump')

    @staticmethod
    def call(args):
        config_file = args.config
        print(args)
        swap_config(config_file)

        fname = args.file
        data = get_metadata(fname)
        upload_data(data)
