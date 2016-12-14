# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import pandas as pd


class Table(object):

	def __init__(self, cat_dist=1.0/6 * np.ones(6)):
		self.categorical_distribution = cat_dist

	# Simulate throwing a dice from table 
	def sample_from_table():
		r = np.random.rand()
		c = np.cumsum(self.categorical_distribution)

		for i in range(6):
			if (r<=c[i]):
				return i+1


class Tables(object):
	self.tables

	def __init__(self, cat_tables):
		tables = [[],[]]
		for cat_table in cat_tables:
			self.tables[0].append(able(cat_tables[0])) # unprimed
			self.tables[1].append(able(cat_tables[1])) # primed


class Casino(object):

	def __init__(self, cat_tables):
		self.tables = Tables(cat_tables)

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

'''f[0][0] =  sum([[cat_distrib(dice_table)]])
f[0][1] = sum([])
for k in range(K):
	f[k][0] = 
	f[k][1]'''

# Sample r_K, r_{K-1}, ..., r_1