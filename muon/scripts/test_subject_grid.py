#!/usr/bin/env python

from muon.utils.subjects import Subjects
import pickle
import os


def main():
    fname = os.path.join(
        os.getenv('MUOND'), 'Data/test_subjects.pkl')
    print('fname: %s' % fname)
    subjects = pickle.load(open(fname, 'rb'))
    sample = subjects._sample_s(100)

    fig = sample.plot_subjects(w=10, grid=True)
    fig.savefig('test.png')


def path(name):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, name)


if __name__ == '__main__':
    main()


