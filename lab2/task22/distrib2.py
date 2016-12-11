import numpy as np
import pylab as plt

p1 = np.ones(6)/6
p2 = np.array([.5, .1, .1, .1, .1, .1])
p3 = np.array([.1, .3, .15, .3, .05, .1])

fig = plt.figure(figsize=(5,10))

figure_title = "Set 1"
ax1  = plt.subplot(1,3,1)

ax1.title(figure_title, fontsize = 20)

figure_title = "Categorical dice distributions"
ax2  = plt.subplot(1,2,2)

plt.text(0.5, 1.08, figure_title,
         horizontalalignment='center',
         fontsize=20,
         transform = ax2.transAxes)

ax1.bar(range(1,7),p1,align='center')
ax2.bar(range(1,7),p2,align='center')

plt.show()