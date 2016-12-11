import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

p1 = np.ones(6)/6
p2 = np.array([.5, .1, .1, .1, .1, .1])
p3 = np.array([0,0,0,0,0,1])

# Enable LaTeX fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig = plt.figure(figsize=(15,4.5))

ax1 = fig.add_subplot(131)
ax1.bar(range(1,7), p1,align='center',color='black')
ax1.set_title(r'Set 1',fontsize=25)

ax2 = fig.add_subplot(132)
ax2.bar(range(1,7), p2,align='center',color='black')
ax2.set_title(r'Set 2',fontsize=25)

ax3 = fig.add_subplot(133)
ax3.bar(range(1,7), p3,align='center',color='black')
ax3.set_title(r'Set 3',fontsize=25)

#fig.suptitle(r"Categorical dice distributions"+"\n"+"\n",
#         horizontalalignment='center',
#         fontsize=20)

fig.tight_layout()#fig.show()

fig.savefig('/Users/lucasrodes/Documents/KTH/ML - adv/HW2/LaTeX/figs/dice_distrib.pdf')
#fig.close()

#