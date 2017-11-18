
from muon.utils.subjects import Subjects
from muon.utils.camera import Camera

import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


class Cluster:

    figure = 0

    def __init__(self, pca, subjects, sample):
        self.pca = pca
        self.subjects = subjects
        self.sample_X = self.project_subjects(sample)

    @classmethod
    def create(cls, subjects, components=8):
        _, charges = cls.scale_charges(subjects)

        pca = PCA(n_components=8)
        pca.fit(charges)

        sample = subjects.sample(1e4)
        return cls(pca, subjects, sample)

    @classmethod
    def run(cls, subjects):
        cluster = cls.create(subjects)
        cluster.plot()

        import code
        code.interact(local=locals())

    def count_class(self, bound, axis, direction):
        s = self.subjects.list()
        _, X = self.project_subjects(s)

        count = 0
        for item in X:
            if direction == 1:
                if item[axis] > bound:
                    count += 1
            else:
                if item[axis] < bound:
                    count += 1

    def visualize(self):
        camera = Camera()
        fig = plt.figure(figsize=(9, 1))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05,
                            wspace=.05)

        count = self.pca.n_components_
        for n in range(count):
            component = self.pca.components_[n, :]
            x, y, c = camera.transform(component)
            ax = fig.add_subplot(1, count+1, n+1, xticks=[], yticks=[])
            ax.scatter(x, y, c=c, s=10, cmap='viridis')

        x, y, c = camera.transform(self.mean_charge())
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        ax.scatter(x, y, c=c, s=10, cmap='viridis')

        plt.show()

    def mean_charge(self):
        subjects = self.subjects.list()
        c = [s.charge for s in subjects]
        c = np.array(c)
        c = np.mean(c, axis=0)
        return c

    def visualize_mean(self):
        camera = Camera()
        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.05,
                            wspace=.05)

        x, y, c = camera.transform(self.mean_charge())
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        ax.scatter(x, y, c=c, s=10, cmap='viridis')

        plt.show()

    def plot_subjects(self, subjects, save=False):
        """
        subjects: list of subject objects
        """
        if subjects is None:
            order, X = self.sample_X
        else:
            order, X = self.project_subjects(subjects)

        sorted_ = {k:[] for k in [-1, 0, 1]}

        for i, s in enumerate(order):
            x, y = X[i, :2]
            label, c = self.subject_plot_label(s)
            sorted_[label].append((x, y, c))

        def plot(v):
            x, y, c = zip(*sorted_[v])
            plt.scatter(x, y, c=c, s=2.5)

        for i in [-1, 0, 1]:
            plot(i)

        plt.axis([-5, 20, -25, 25])
        plt.title('PCA dimensionality reduction of Muon Data')
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')

        # c = [s.label for s in subjects]
        if save:
            plt.savefig('Figure_%d' % self.figure)
            self.figure += 1
        else:
            plt.show()

    def subject_plot_label(self, subject_id):
        s = self.subjects[subject_id]
        return s.label, s.color()

    def plot(self, save=False):
        self.plot_subjects(None, save=save)

    def plot_class(self, class_):
        if type(class_) is int:
            class_ = [class_]
        subjects = [s for s in self.subjects.list() if s.label in class_]

        if len(subjects) > 1e4:
            subjects = random.sample(subjects, 1e4)

        self.plot_subjects(subjects)

    def download_plotted_subjects(self, x, y, c, size, prefix='', dir_=None):
        subjects = self.subjects_in_range(x, y, c, self.sample_X[0])
        self.download_subjects(subjects, size, prefix, dir_)

    def subjects_in_range(self, x, y, c, subjects=None):
        if subjects is None:
            subjects = self.subjects.list()

        # Remap bounding box coordinates so name doesn't conflict
        x_ = x
        y_ = y

        if type(c) is int:
            c = [c]

        def in_range(x, y):
            """
            Check if coordinates are inside the bounding box
            """
            def check(x, bounds):
                if bounds is None:
                    return True

                min, max = bounds
                if x > min and x < max:
                    return True
                return False

            return check(x, x_) and check(y, y_)

        def check_type(subject):
            """
            Check if subject is in the required class
            """
            if c is None:
                return True
            subject = self.subjects[subject]
            return subject.label in c

        order, X = self.project_subjects(subjects)
        subjects = []
        for i, point in enumerate(X):
            x, y = point
            if in_range(x, y):
                subject = order[i]
                if check_type(subject):
                    subjects.append(subject)

        return subjects

    def download_subjects(self, subjects, size=None, prefix='', dir_=None):
        """
        Download subject images from panoptes

        subjects: list of subject ids
        size: select random sample from list of subjects
        """
        if size is not None and size > len(subjects):
            subjects = random.sample(subjects, size)
        subjects = [self.subjects[s] for s in subjects]
        Subjects(subjects).download_images(prefix, dir_)

    def subject_images(self, subjects, size=None):
        """
        Get list of scikit-image objects of subject images from panoptes

        subjects: list of subject ids
        size: select random sample from list of subjects
        """
        print(len(subjects), size)
        if size is not None and size < len(subjects):
            subjects = random.sample(subjects, size)
        print(len(subjects))
        subjects = [self.subjects[s] for s in subjects]
        return Subjects(subjects).load_images()

    @staticmethod
    def get_charge(subject):
        return np.array(subject.charge)

    def project_subjects(self, subjects):
        """
        subjects: list of subjects to project
        """
        charges = np.zeros
        order, charges = self.build_charge_array(subjects)
        X = self.pca.transform(charges)
        return order, X

    @classmethod
    def scale_charges(cls, subjects):
        """
        subjects: subjects object
        """
        subject_order = []
        _subjects = subjects.list()
        charges = np.zeros((len(_subjects), len(_subjects[0].charge)))
        for i, subject in enumerate(_subjects):
            subject_order.append(subject.id)
            charges[i] = cls.get_charge(subject)

        charges = preprocessing.scale(charges)
        print(charges)
        subjects.scale_charges(subject_order, charges)

        return subject_order, charges

    @classmethod
    def build_charge_array(cls, subjects):
        subject_order = []
        charges = np.zeros((len(subjects), len(subjects[0].scaled_charge)))
        for i, subject in enumerate(subjects):
            subject_order.append(subject.id)
            charges[i] = subject.scaled_charge

        return subject_order, charges






    

"""
1. open a hdf5 file
2. iterate through events in file
    filter out duplicate events/telescopes
3. find appropriate subject
4. build data array
5. run pca
6. plot results"""
