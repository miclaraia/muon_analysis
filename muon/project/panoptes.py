
from panoptes_client.project import Project
from panoptes_client.panoptes import Panoptes
from panoptes_client.subject_set import SubjectSet
from panoptes_client.subject import Subject
from panoptes_client.panoptes import PanoptesAPIException

import math
import logging
logger = logging.getLogger(__name__)


class Uploader:
    _client = None

    def __init__(self, project, group):
        self.client()
        self.project = self.get_project(project)
        self.subject_set = self.get_subject_set(group)
        self.subject_queue = []

    @classmethod
    def client(cls):
        if cls._client is None:
            cls._client = Panoptes.connect(login='interactive')
        return cls._client

    @staticmethod
    def get_project(project):
        return Project.find(project)

    def get_subject_set(self, group):
        project = self.project
        name = 'Auto_Group_%d' % group

        for subject_set in project.links.subject_sets:
            print(subject_set)
            if subject_set.display_name == name:
                return subject_set
        logger.debug('Project: %s', self.project)
        print(self.project)
        subject_set = SubjectSet()

        subject_set.links.project = project
        subject_set.display_name = name

        subject_set.save()
        return subject_set

    def get_subjects(self):
        logger.debug([s for s in self.subject_set.subjects])
        return [(s.id, s.metadata['id']) for s in self.subject_set.subjects]

    def add_subject(self, subject):
        subject.links.project = self.project

        try:
            subject.save()
        except PanoptesAPIException as e:
            logger.info('Cleaning up')
            logger.warn('Removing subjects: %s' % str(self.subject_queue))
            for subject in self.subject_queue:
                Subject.delete(
                    subject.id, headers={'If-Match': subject.etag})
            raise e

        self.subject_queue.append(subject)

        print(subject)
        return subject

    def unlink_subjects(self, subjects, delete=True):
        """
        Delete subjects from a subject set

        subjects: list of zooniverse subject ids
        """
        logger.info('Getting existing subjects')
        subject_set = self.subject_set
        subjects = [s for s in subject_set.subjects if s.id in subjects]
        logger.info('Unlinking subjects')
        print(subjects)
        subject_set.remove(subjects)
        subject_set.save()

        # TODO this doesn't actually work
        # if delete:
            # for s in subjects:
                # Subject.delete(s.id, headers={'If-Match': s.etag})

    def upload(self):
        print('Linking %d subjects to subject set %s' %
              (len(self.subject_queue), self.subject_set))
        subjects = self.subject_queue
        l = len(subjects)
        for i in range(math.ceil(l/1000)):
            a = i*1000
            b = min((i+1)*1000, l)
            self.subject_set.add(subjects[a:b])
            self.subject_set.save()

        self.subject_queue = []
