#!/usr/bin/env python

import panoptes_client as pan
import code


def main():
    client = pan.Panoptes(login='interactive')

    c = load({})
    d = load({'created_at':'2018-01-29T04:38:00.014Z'})
    code.interact(local={**globals(), **locals()})


def load(scope):
    scope.update({'project_id': 5918})
    c = pan.Classification.where(scope='project', **scope)
    print(scope)
    return [i for i in c]


if __name__ == '__main__':
    main()
