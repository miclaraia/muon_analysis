#!/usr/bin/env python

from muon.project.images import Random_Images

def main():
    images = Random_Images.load_group(2)
    im = images.images[767]
    print('subject: ', im.at_location(991, 1860))


main()
