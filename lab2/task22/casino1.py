# -*- coding: utf-8 -*-
import numpy as np
from pprint import pprint
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

# Generation of a number
def throw_dice(value):
	# Fair
	if(value == 0):
		return np.random.randint(1,7)
	# Unfair 1
	elif(value == 1):
		r = np.random.rand()
		if(r <= 0.5):
			return 1
		elif(r <= 0.6):
			return 2
		elif(r <= 0.7):
			return 3
		elif(r <= 0.8):
			return 4
		elif(r <= 0.9):
			return 5
		else:
			return 6
	# Unfair 2
	else:
		return 6
		'''r = np.random.rand()
		if(r <= 0.3):
			return 2
		elif(r <= 0.6):
			return 4
		elif(r <= 0.7):
			return 1
		elif(r <= 0.85):
			return 3
		elif(r <= 0.9):
			return 5
		else:
			return 6'''

# Number of tables visited
K = 10
#dice_table = np.zeros([2, K]) # case 1, 2
dice_table = np.vstack([np.zeros([1, K]), 2*np.ones([1, K])]) # case 3

# Number of players
N = 4
#dice_player = np.zeros(N) # case 1
dice_player = np.array([0,0,1,1]) # case 2, 3

S = []

for n in range(N):
	S.append([])
	if (np.random.rand() >= 0.5):
		id_table = 0
	else:
		id_table = 1
	for k in range(K):
		# Dice of table
		X = throw_dice(dice_table[id_table, k])
		# Dice of player
		Z = throw_dice(dice_player[n])
		# Sum and add to the corresponding S
		S[n].append(X+Z)
		# Change to a primed table?
		if (np.random.rand() <= 0.75):
			id_table  = 1-id_table

# Enable LaTeX fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

p = []
print "preparing plots..."
fig = plt.figure(figsize=(10,5))
#plt.title(r'Histogram of $S_k^i$')
for n in range(N):
	ax = fig.add_subplot(2,2,n+1)
	#sns.distplot(S[n], range(2,14))
	p = np.histogram(S[n], range(2,14))
	pp = p[0]/float(sum(p[0]))
	ax.bar(p[1][:-1], pp,color='black',align='center')
	ax.set_title(r'Player '+str(n),fontsize=17)
	#pd.Series(S[n]).hist()
	p = np.bincount(S[n])
	#p = p/float(sum(p))
	for i in range(1,len(p)):
		print "%.3f" % p[i],
	print ""

# AFEGIR COLUMNE AL DATAFRAME AMB STRING "BIASED", "UNBIASED", 
# + MODIFICAR VIOLINPLOT, FER US DE HUE AMB LATRIBUT AFEGIT!

plt.figure(2)
s = pd.DataFrame(np.transpose(S))
#s.boxplot()
#sns.violinplot(data=s, palette="Set3", bw=.2, cut=1, linewidth=1)
#sns.plt.title(r'Violinplot of $S_k^i$ for all users')
plt.show()
fig.tight_layout()#fig.show()
fig.savefig('/Users/lucasrodes/Documents/KTH/ML - adv/HW2/LaTeX/figs/case3_casino.pdf')

