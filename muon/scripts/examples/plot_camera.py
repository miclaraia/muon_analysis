from muon.subjects.subject import Subject
from muon.subjects.subjects import Subjects
import matplotlib.pyplot as plt
import numpy as np


charge = np.array(range(499))
s = Subject(None, charge)

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)

s.plot(ax)

fig.savefig('test.png')

subjects = Subjects([Subject(i, charge) for i in range(10)])
fig = subjects.plot_subjects()
fig.savefig('test2.png')



