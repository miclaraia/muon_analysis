#!/usr/bin/env python

import click
import os

from panoptes_client.project import Project
from panoptes_client.panoptes import Panoptes
from panoptes_client.subject_set import SubjectSet
from panoptes_client.subject import Subject
from panoptes_client.panoptes import PanoptesAPIException


@click.group(invoke_without_command=True)
def main():
    client = Panoptes.connect(
        login='interactive',
        endpoint='https://panoptes-staging.zooniverse.org')

    project = Project.find(1815)
    subject_set = SubjectSet()
    subject_set.links.project = project
    subject_set.display_name = 'Test subject_set'
    subject_set.save()


    import code
    code.interact(local={**globals(), **locals()})


if __name__ == '__main__':
    main()
