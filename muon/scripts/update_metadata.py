#!/usr/bin/env python

import muon.project.panoptes as pan
from muon.project.images import Images, Random_Images


def main():
    images = Images.load_group(0)
    uploader = pan.Uploader(5918, 0)
    print('Updating metadata')
    
    for subject in uploader.subject_set.subjects:
        print(subject)
        image = images.images[subject.metadata['id']]
        print(image)

        print(subject.metadata)
        subject.metadata = image.dump_manifest()
        print(subject.metadata)

        subject.save()
    print('done')


if __name__ == '__main__':
    main()
            
