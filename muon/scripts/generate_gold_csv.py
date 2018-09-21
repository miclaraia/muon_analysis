#!/usr/bin/env python

"""
Script to generate gold labels csv file from classifications made by experts
on the Zooniverse.
"""

import csv
import muon.project.parse_export as pe
import muon.data as data


def main():
    agg = pe.Aggregate.load('mh2_gold')
    print(agg)
    print(agg.data['subjects'])
    print(agg.reduce()[1])
    print(agg.subject_labels())

    with open(data.path('mh2_golds.csv'), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['subject', 'gold'])
        for s, g in agg.subject_labels().items():
            writer.writerow([s, g])




if __name__ == '__main__':
    main()
