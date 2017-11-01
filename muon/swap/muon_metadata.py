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



def swap_config(config):
    swap.config.import_config(config)
    swap.config.logger.init()


def get_path():
    p = os.path
    path = p.dirname(p.abspath(__name__))
    return path


class MuonMetadata:

    @staticmethod
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


class SubjectID(MuonMetadata):

    regex = re.compile('|'.join([
        '[a-z_]*(?:run)?([0-9]+)[a-z_]*evt([0-9]+)(?:_tel[0-9])?.jpeg',
    ]))

    @classmethod
    def run(cls, config_file, fname):
        swap_config(config_file)

        data = cls.collect_data(fname)
        cls.upload_data(data)

    @classmethod
    def test_regex(cls, config_file, fname):
        swap_config(config_file)

        for i in cls.get_data(fname):
            pass

    ###########################################################################
    #####   ###################################################################
    ###########################################################################

    @classmethod
    def collect_data(cls, fname):
        data = {}
        for subject, run, evt in cls.get_data(fname):
            data[subject] = (run, evt)

        return data

    @classmethod
    def get_data(cls, fname):
        with open(fname, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                yield cls.parse_row(row)


    @classmethod
    def parse_row(cls, row):
        meta = json.loads(row['metadata'])

        subject = int(row['subject_id'])

        # random exception where the filename wasn't encoded into the
        # subject dump. I think this subject is the only exception
        if subject == 6396050:
            return (subject, 78596, 275237)

        fname = None
        for value in meta.values():
            if type(value) is str and 'jpeg' in value:
                fname = value

        if fname is None:
            print(meta)
            raise Exception('Couldn\'t find filename in meta field')

        print('subject %d fname %s' % (subject, fname))

        run_, evt = cls.parse_fname(fname)

        metadata = subject, run_, evt
        print(metadata)

        return metadata

    @classmethod
    def parse_fname(cls, fname):
        m = cls.regex.match(fname)
        if not m:
            print(fname)
            raise Exception
        run_, evt = m.groups()
        run_ = int(run_)
        evt = int(evt)
        return run_, evt
