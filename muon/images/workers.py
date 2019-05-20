import os
import logging
from tqdm import tqdm
import numpy as np

from muon.config import Config
from muon.images.image import Image
from muon.images.image_group import ImageGroup
from muon.subjects.storage import Storage
import muon.project.panoptes as panoptes

logger = logging.getLogger(__name__)


class Workers:

    def __init__(self, database):
        self.database = database

    def create_jobs(self, image_ids, job_type):
        with self.database.conn as conn:
            _next_id = self.database.ImageWorker.next_id(conn)

            def next_id_wrapper():
                i = 0
                while True:
                    yield _next_id + i
                    i += 1
            next_id_iter = next_id_wrapper()

            def next_id():
                return next(next_id_iter)

            image_ids = np.array(image_ids)
            batches = np.array_split(image_ids, max(1, image_ids.size/100))
            print(batches)

            jobs = []
            for batch in batches:
                job = Job(next_id(), batch, job_type, 0)
                jobs.append(job)

            self.database.ImageWorker.add_jobs(conn, jobs)
            conn.commit()

    def generate_images(self, group_id):
        with self.database.conn as conn:
            image_ids = list(
                self.database.Image.get_group_image_ids(
                    conn, group_id, shuffle=True))
        self.create_jobs(image_ids, 'generate')

    def upload_images(self, group_id):
        with self.database.conn as conn:
            image_ids = list(
                self.database.Image.get_group_image_ids(
                    conn, group_id, exclude_zoo=True, shuffle=True))
        self.create_jobs(image_ids, 'upload')

    def clear_jobs(self):
        with self.database.conn as conn:
            self.database.ImageWorker.clear_jobs(conn)

    def run(self):
        with self.database.conn as conn:
            job = self.database.ImageWorker.get_job(conn)
            if job is None:
                return False

            job = Job(**job)
            self.database.ImageWorker.set_job_status(conn, job.job_id, 1)
            conn.commit()

        job.run(self.database)
        with self.database.conn as conn:
            self.database.ImageWorker.set_job_status(conn, job.job_id, 2)
            conn.commit()

        return True

    def run_all(self):
        while self.run():
            pass


class Job:

    def __init__(self, job_id, image_ids, job_type, job_status):
        self.job_id = job_id
        self.image_ids = image_ids
        self.job_type = job_type
        self.job_status = job_status

    @classmethod
    def from_db(cls, kwargs):
        image_ids = kwargs['image_ids'].split(',')
        image_ids = [int(i) for i in image_ids]
        kwargs['image_ids'] = image_ids

        return cls(**kwargs)

    def generate(self, images, database):
        subject_storage = Storage(database)
        config = Config.instance()
        image_path = config.storage.images
        dpi = config.plotting.dpi

        group_ids = []

        for image, image_width in tqdm(images):
            group_id = image.group_id
            image_path = os.path.join(
                config.storage.images, 'group_%d' % group_id)
            if group_id not in group_ids:
                group_ids.append(group_id)

                logger.debug('image_path: %s', image_path)
                if not os.path.isdir(image_path):
                    os.mkdir(image_path)

            if image.generate(image_width, subject_storage,
                              dpi=dpi, path=image_path):
                logger.info(image)

    def upload(self, images, database):
        config = Config.instance()
        image_path = config.storage.images
        project_id = config.panoptes.project_id

        uploaders = {}

        print('Creating Panoptes subjects')
        for image, _ in tqdm(images):
            if image.zoo_id is not None:
                continue

            group_id = image.group_id
            group = ImageGroup(group_id, database)
            if group_id not in uploaders:
                uploader = panoptes.Uploader(
                    project_id, group=group_id,
                    subject_set=group.zoo_subject_set)

                if group.zoo_subject_set is None:
                    group.zoo_subject_set = uploader.subject_set.id
                    group.save()
                uploaders[group_id] = uploader

            uploader = uploaders[group_id]

            fname = os.path.join(
                image_path, 'group_%d' % group_id, image.fname())

            subject = panoptes.Subject()
            subject.add_location(fname)
            subject.metadata.update(image.dump_manifest())

            subject = uploader.add_subject(subject)
            image.zoo_id = subject.id

            logger.info(image)

    def run(self, database):

        with database.conn as conn:
            images = list(database.ImageWorker
                          .get_job_images(conn, self.job_id))

        def image_iter():
            for image_dict, image_width, image_type in images:
                image = Image.create_image(image_type,
                    image_dict['image_id'], database,
                    online=True, attrs=image_dict)
                yield image, image_width

        if self.job_type == 'upload':
            self.upload(image_iter(), database)
        elif self.job_type == 'generate':
            self.generate(image_iter(), database)


