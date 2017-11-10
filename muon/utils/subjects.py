import panoptes_client as pclient
import urllib.request
import os


def get(subject_id):
    return pclient.subject.Subject(subject_id)

def get_image_url(subject_id):
    s = get(subject_id)
    url = list(s.raw['locations'][0].values())[0]
    return url

def download_image(subject_id, prefix=None, dir_=None):
    if prefix is None:
        prefix = ''

    url = get_image_url(subject_id)
    ext = url.split('.')[-1]

    fname = 'muon-%s-%d.%s' % (prefix, subject_id, ext)
    if dir_ is not None:
        fname = os.path.join(dir_, fname)

    urllib.request.urlretrieve(url, fname)

def download_images(subjects, prefix=None, dir_=None):
    for i in subjects:
        download_image(i, prefix, dir_)
