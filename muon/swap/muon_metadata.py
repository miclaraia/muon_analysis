#!/usr/bin/env python

from swap.db import DB
import swap.config
from muon.ui import ui

from tqdm import tqdm
import argparse
import csv
import json
import re
import os
import sys

import logging
logger = logging.getLogger(__name__)


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

        i = 0
        print(i)
        for subject, metadata in data.items():
            r = db.subjects.update_metadata(subject, metadata, False)
            requests.append(r)

            if i % 10000:
                sys.stdout.flush()
                sys.stdout.write('%d\r' % i)
            if len(requests) > 1e5:
                write()
                requests = []

            i += 1
        write()


class SubjectID(MuonMetadata):

    regex = re.compile('|'.join([
        '[a-z_]*(?:run)?([0-9]+)[a-z_]*evt([0-9]+)(?:_tel([0-9]))?.jpeg',
    ]))

    @classmethod
    def run(cls, fname):
        data = cls.collect_data(fname)
        cls.upload_data(data)

    @classmethod
    def test_regex(cls, fname):
        subjects = {}
        evts = {}
        for subject, evt in cls.get_data(fname):
            if subject not in subjects:
                subjects[subject] = [evt]
            else:
                subjects[subject].append(evt)

            if evt not in evts:
                evts[evt] = [subject]
            else:
                evts[evt].append(subject)

        subjects = {k:v for k, v in subjects.items() if len(v) > 1}
        evts = {k:v for k, v in evts.items() if len(v) > 1}
        from pprint import pprint
        pprint(subjects)
        pprint(evts)

        print('subjects %d evts %d' % (len(subjects), len(evts)))


    ###########################################################################
    #####   ###################################################################
    ###########################################################################

    @classmethod
    def collect_data(cls, fname):
        data = {}
        for subject, evt in tqdm(cls.get_data(fname)):
            metadata = {
                'run': evt[0],
                'evt': evt[1],
                'tel': evt[2],
            }
            if subject not in data:
                data[subject] = metadata

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
            return (subject, (78596, 275237, -1))

        fname = None
        for value in meta.values():
            if type(value) is str and 'jpeg' in value:
                fname = value

        if fname is None:
            print(meta)
            raise Exception('Couldn\'t find filename in meta field')

        # print('subject %d fname %s' % (subject, fname))

        evt = cls.parse_fname(fname)

        metadata = subject, evt
        #print(metadata)

        return metadata

    @classmethod
    def parse_fname(cls, fname):
        m = cls.regex.match(fname)
        if not m:
            print(fname)
            raise Exception
        run_, evt, tel = m.groups()
        run_ = int(run_)
        evt = int(evt)

        if tel is None:
            tel = -1
        else:
            tel = int(tel)
        return run_, evt, tel

