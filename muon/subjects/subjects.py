
from collections import OrderedDict
import os
import re
import numpy as np
import h5py
import random
import math
import progressbar
import matplotlib.pyplot as plt
import string

from muon.utils.camera import Camera, CameraRotate
from muon.subjects.subject import Subject


class Subjects:

    _mapping = None

    def __init__(self, subjects):
        if type(subjects) is list:
            subjects = [(s.id, s) for s in subjects]
        self.subjects = OrderedDict(subjects)

    def sample(self, size):
        size = int(size)
        subjects = list(self.subjects.values())
        print('number of subjects', len(subjects))
        if size > len(subjects):
            return subjects
        return random.sample(subjects, size)

    def get_subject(self, s):
        if '_' in s:
            s, n = s.split('_')
            n = int(n)
            return self.subjects[s].copy(n)
        return self.subjects[s]

    def get_dimensions(self):
        return len(self.subjects), len(next(self.iter()).x)


    ##########################################################################
    ###   Plotting   #########################################################
    ##########################################################################

    def plot_sample(self, size, **kwargs):
        s = self._sample_s(size)
        return s.plot_subjects(plt.figure(), **kwargs)

    def plot_subjects(self, fig=None, w=5, camera=None,
                      grid=False, grid_args=None, meta=None):
        if camera is None:
            camera = Camera()

        if fig is None:
            fig = plt.figure()

        l = math.ceil(len(self.subjects) / w)
        fig.set_size_inches(2*w, 2*l)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05,
                            wspace=.05)

        axes = fig.subplots(l, w, True, subplot_kw={
            'xticks': [],
            'yticks': []
        })
        if l > 1:
            axes = axes.flatten()
        if w == 1:
            axes = [axes]
        for i, subject in enumerate(self.list()):
            subject.plot(axes[i], camera)

        if grid:
            if grid_args is None:
                grid_args = {}
            fig = self._plot_add_grid(fig, w, l, **grid_args)


        if meta:
            size = fig.get_size_inches()
            meta = {
                'height': size[1],
                'width': size[0],
                'rows': l,
                'cols': w,
            }

            return fig, meta
        return fig

    @staticmethod
    def _plot_add_grid(fig, w, l, offset=None):

        # Calculate the offset in x and y directions
        # offset initially in inches, then offset_x,y are fractional
        if offset is None:
            offset = .5
        width, height = fig.get_size_inches()
        width += offset
        height += offset
        fig.set_size_inches(width, height)

        offset_x = offset/width
        offset_y = offset/height

        # Add the subplot to draw the grid and text
        ax = [offset_x, 0, 1-offset_x, 1-offset_y]
        ax = fig.add_axes(ax, facecolor=None, frameon=False)
        fig.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add the row components (horizontal lines and row index)
        for i in range(l):
            y = (i+1)/l
            if y < 1:
                ax.plot([0, 1], [y, y], c='black', alpha=.6)

            # Calculate text location
            x = offset_x/2
            y = (y-1/2/l)*(1-offset_y)
            fig.text(x, y, str(l-i), fontsize=14, transform=fig.transFigure)

        # Add the column components (vertical lines and column letters)
        for i in range(w):
            x = (i+1)/w
            if x < 1:
                ax.plot([x, x], [0, 1], c='black', alpha=.6)

            # Calculate text location
            #x = ((x-1/2/w)*(1-offset_x)+(2/3*offset_x))
            x = (2*i+1)/2/w*(1-offset_x)+.02
            y = 1-offset_y*2/3
            a = string.ascii_uppercase[i]
            fig.text(x, y, a, fontsize=14, transform=fig.transFigure)

        fig.subplots_adjust(left=offset_x, top=(1-offset_y))
        return fig

    ##########################################################################
    ###   Subject Charge Data   ##############################################
    ##########################################################################

    def get_xy(self, subjects, label):
        x = np.zeros((len(subjects), self.get_dimensions()[1]))
        y = np.zeros((len(subjects,)))
        for i, s in enumerate(subjects):
            s = self.get_subject(s)
            x[i,:] = s.x
            y[i] = s.y[label]

        return x, y

    def get_x(self):
        x = np.zeros(self.get_dimensions())
        for i, subject in enumerate(self.iter()):
            x[i,:] = subject.x
        return x

    ##########################################################################
    ###   Operator Overloading   #############################################
    ##########################################################################

    def __getitem__(self, subject):
        return self.get_subject(subject)

    def __len__(self):
        return len(self.subjects)

    def __str__(self):
        return '%d subjects' % len(self.subjects)

    def __repr__(self):
        return str(self)

    ##########################################################################
    ###   Subsets   ##########################################################
    ##########################################################################

    def _sample_s(self, size):
        return self.__class__(self.sample(size))

    def iter(self):
        for s in self.subjects.values():
            yield s

    def list(self):
        return list(self.subjects.values())

    def keys(self):
        return list(self.subjects.keys())

    def subset(self, subjects):
        subset = [self.get_subject(s) for s in subjects]
        return self.__class__(subset)

    def labeled_subjects(self):
        return self.subset([s.id for s in self.iter() if s.y in [0, 1]])

    def apply_labels(self, labels):
        for subject_id, label in labels:
            self.subjects[subject_id].y = label

    @property
    def shape(self):
        l = len(self.subjects)
        if l > 0:
            w = len(next(self.iter()).x)
            return l, w
        return l,

    @classmethod
    def evt_to_subj(cls, evt, mapping):
        """
        Get subject associated with specific run event and telescope

        evt: (run, evt, tel)
        """
        if evt in mapping:
            return mapping[evt]

        # Event not in mapping. Try again with neutral telescope
        evt = (*evt[:2], -1)
        return mapping.get(evt, None)


