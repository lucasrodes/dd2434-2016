# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.spatial.distance as dd
import sys
import pandas as pd

# Agent can be a table or a player
class Agent(object):

	def __init__(self, cat_dist):
		self.categorical_distribution = cat_dist
		self.cdf = np.cumsum(self.categorical_distribution)

	# Simulate throwing a dice from table/player
	def sample(self):
		r = np.random.rand()
		#print self.cdf
		for i in range(6):
			if (r<=self.cdf[i]):
				return i+1


# All tables in the casino
class Tables(object):

	def __init__(self, cat_tables):
		self.K = len(cat_tables[0])
		self.tables = [[],[]]
		for k in range(K):
			self.tables[0].append(Agent(cat_tables[0][k])) # unprimed
			self.tables[1].append(Agent(cat_tables[1][k])) # primed

	def sample_from_table(self, k, is_primed):
		return self.tables[is_primed][k].sample()

	def cat_from_table(self, k, is_primed):
		return self.tables[is_primed][k].categorical_distribution

# All players taking part in the game
class Players(object):

	def __init__(self, cat_players):
		self.players = []
		for cat_player in cat_players:
			self.players.append(Agent(cat_player))
		self.N = len(self.players)

	def sample_from_player(self, n):
		return self.players[n].sample()

	def get_list(self):
		return self.players


# Casino itself, with tables
class Casino(object):

	def __init__(self, cat_tables, init_prob, trans_prob):
		#print cat_tables[0]
		#print cat_tables[1]
		self.tables = Tables(cat_tables)
		self.K = len(cat_tables[0])
		self.pi = init_prob
		self.A = trans_prob

	def play(self, players):
		S = []
		R = []
		for n in range(players.N):
			S.append([])
			R.append([])
			r = np.random.binomial(1, pi[1])
			for k in range(K):
				# Dice of table
				X = self.tables.sample_from_table(k, r)
				# Dice of player
				Z = players.sample_from_player(n)#player.sample()
				# Sum and add new observation
				S[n].append(X+Z)
				# Add new visited table
				R[n].append(r)
				# p(change) = 3/4, p(remain) = 1/4 
				r = np.random.binomial(1, A[r][1])
		return S, R

	def prob_observation_at_table(self, s, k, is_primed):
		return sum([(x+z == s)*self.tables.cat_from_table(k,is_primed)[x-1]
			*player.categorical_distribution[z-1] for x in range(1,7) for z in range(1,7)])

	def obtain_f(self, S, player):
		f = np.zeros([self.K, 2])
		f[0] =  [pi[r]*self.prob_observation_at_table(S[0], 0, r) for r in range(2)]

		for k in range(1,K):
			f[k] =  [self.prob_observation_at_table(S[k], k, r)*sum([f[k-1][j]*self.A[j][r] 
				for j in range(2)]) for r in range(2)]
		return f


# Initial parameters
K = 200 # Number of visited tables per player
N = 1 # Number of players

# Categorical distribution for tables
cat_tables_unprimed = [1.0/6 * np.ones(6) for k in range(K)]  # fair dice for unprimed tables
cat_tables_primed = [[0, 0, 0, 0, 0, 1] for k in range(K)]  # unfair dice for primed tables
cat_tables = [cat_tables_unprimed, cat_tables_primed]
# Categorical distribution for players
cat_players = [ 1.0/6 * np.ones(6) for n in range(N)] # fair dice for players
# Initial table distribution
pi = [0, 1]#[.5, .5]
# Table transition matrix
A = [[.9,.1],[.7, .3]]#[[.25, .75],[.75, .25]]
#A = [[0, 1],[1, 0]]

# Initialize players
players = Players(cat_players)

# Generate observation
casino = Casino(cat_tables, pi, A)
S, R = casino.play(players)
player = players.get_list()[0]
#S = [4,5,4] # for debugging
f = casino.obtain_f(S[0], player) # we only use first player

# sample r_K
r = np.zeros(K).astype(int)
F = f[K-1][1]/(f[K-1][0] + f[K-1][1])
r[K-1] = np.random.binomial(1, F)
# sample the rest
for k in range(K-2,-1,-1):
	F = (A[1][r[k+1]]*f[k][1])/(A[0][r[k+1]]*f[k][0]+A[1][r[k+1]]*f[k][1])
	r[k] = np.random.binomial(1, F)

#columns=range(K),
listt = [S[0], R[0], r.tolist()]
df = pd.DataFrame(listt,index=['Observations','True Sequence','Estimation'])
#df.append(pd.Series(r),ignore_index=True)
#print df.to_csv(sep='\t',header=False)
#print "Observations  :", np.array(S[0])
#print "Table Sequence:", np.array(R[0])
#print "f parameter:"
#print f
#print "Estimation    :", r
#print ""
print "Error in r    :", dd.cityblock(r,R[0])/float(len(r))

plt.plot(R[0])
plt.plot(r)
plt.show()
# Sample r_K, r_{K-1}, ..., r_1