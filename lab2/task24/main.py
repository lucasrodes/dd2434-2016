# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import pandas as pd


def throw_dice(value):
	# Fair
	if(value == 0):
		return np.random.randint(1,7)
	# Unfair 1
	else:
		return np.random.randint(1,3)

def cat_distrib(value):
	if(value == 0):
		return 1.0/6 * np.ones(6)
	# Unfair 1
	else:
		return 1.0/2 * [1,1,0,0,0,0]

# Initial parameters
K = 10
dice_table = np.vstack([np.zeros([1, K]), np.ones([1, K])]) # unprimed fair, primed unfair
N=1
dice_player = 0 # fair dice for player
pi = [.5,.5] # Initial state ditribution

visited_tables = []
S = []

# Generate observations
if (np.random.rand() >= 0.5):
	id_table = 0
else:
	id_table = 1
for k in range(K):
	visited_tables.append(id_table)
	# Dice of table
	X = throw_dice(dice_table[id_table, k])
	# Dice of player
	Z = throw_dice(dice_player)
	# Sum and add to the corresponding S
	S.append(X+Z)
	# Change to a primed table?
	if (np.random.rand() <= 0.75):
		id_table  = 1-id_table

print "Observations:", S
print "True table sequence:",visited_tables

# Obtain f_k(z_k), matrix K x 2
f = np.zeros([K, 2])

f[0][0] =  sum([[]])
f[0][1] = sum([])
for k in range(K):
	f[k][0] = 
	f[k][1]

# Sample r_K, r_{K-1}, ..., r_1