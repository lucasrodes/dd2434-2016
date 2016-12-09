# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
#import pylab as pb
#from matplotlib import rc
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import random


import  numpy  as np
import  scipy  as sp
import  scipy.optimize  as opt

import itertools as it
from math import exp, sqrt, pi,cos,sin
import scipy.stats

#from assignment_01_code_index import create_index_set
from create_index_set import create_index_set
import matplotlib.pyplot as pb
import pickle

# Prior of the parameters, using a gaussian distribution N(mu, sigma).
# We assume that all parameters are independent with  have the same parameter mu, and same sigma.
def prior_parameters(mu, sigma, n):
	return np.random.normal(mu, sigma, [3, n])

S = int(1000)
# Pick some theta from the prior P(theta)
Mu = np.array([0,0,0])
Mu_n = np.array([5,5,5])
Sigma = sqrt(1000)*np.eye(3)

# Define unitary matrix and make it have power Sigma
alpha = 2
Sigma_nd = sqrt(1000)*np.array([[0.8, 0.2, 0],[0.2, 0.7, 0.1],[0, 0.1, 0.9]])
#np.array([[cos(alpha),sin(alpha),0],[-sin(alpha),cos(alpha),0],[0,0,1]])
#Sigma_nd *= sqrt(1000)

Theta = np.transpose(np.random.multivariate_normal(Mu_n,Sigma_nd,S))
#Theta = prior_parameters(0, sqrt(1000), S) # 3 x S

with open('Theta.pickle', 'w') as f:
 	pickle.dump(Theta, f)
f.close()

#with open('Theta.pickle') as f:
#	Theta = pickle.load(f)
#f.close()

# Given a dataset, draw the corresponding grid
def drawGrid(dataset):
	print " ___________"
	for i in range(3):
		print "    |   |  "
		print " ",
		for j in range(2):
			print "X |" if (d[j+3*i]==-1) else "O |",
		print "X" if (d[2+3*i]==-1) else "O"
		print " ___|___|___"
	print""


# Return the evidence of a dataset given a model and its parameters theta
def evidence_model_theta(dataset, theta, model):
	
	# Uniform model
	#print theta[0]
	#print "------"
	if model == 0:
		return 1./512
	# Logistic regression, only considers x_1
	elif model == 1:
		#return np.prod([theta[0] for n in range(len(dataset))])
		return np.prod([1./(1+np.exp(-dataset[n]*theta[0]*(n%3-1))) 
			for n in range(9)])
	# Logistic regression, considers both x_1 and x_2
	elif model == 2:
		return np.prod([1./(1+np.exp(-dataset[n]*(theta[0]*(n%3-1)+theta[1]*(1-n/3)))) 
			for n in range(9)])
	# Considers x_1, x_2 and a bias term
	elif model == 3:
		return np.prod([1./(1+np.exp(-dataset[n]*(theta[0]*(n%3-1)+theta[1]*(1-n/3) + theta[2]))) 
			for n in range(9)])
	# No model for this index
	else:
		print "non-valid model index"


# Return the evidence of a dataset given the different models and the parameters theta
def evidence_models_theta(dataset, theta):
	p = [-1, -1, -1, -1]
	for i in range(4):
		p[i] = evidence_model_theta(dataset, theta, i)
		# print "Probability model " + str(i) +" =" + str(p)
	return p


# Return the evidence of a dataset given a model
def evidence_model(dataset, model):
	p = np.mean([evidence_model_theta(dataset, Theta[:,i], model) for i in range(S)])
	return p


# Generate the locations
x_possible = [-1, 0, 1]
x = np.array(list(it.product(x_possible, x_possible))) # 9 x 2 vector

# Generate the Datasets
D = np.array(list(it.product([-1,1],repeat=9))) # 512 x 9
N_D = np.size(D,0) # cardinality of D

# Number of models
N_M = 4

# Display some dataset
'''i = 511
d = D[i]
print ""
print "Dataset ", str(i) + ":"
drawGrid(d)'''

# theta = prior_parameters(mean, std, 1)


evidence = np.zeros([N_M,N_D])
for i_m in range(N_M):
	for i_d in range(N_D):
		evidence[i_m, i_d] = evidence_model(D[i_d], i_m)

#print "evidence"

#print "3m", D[156]

# Which Datasets are the ones each model allocates the most of their probability mass?
D_max = np.argmax(evidence,axis=1)
D_min = np.argmin(evidence,axis=1)

print "Maximums:", D_max
print "0", D[D_max[0]]
print "1", D[D_max[1]]
print "2", D[D_max[2]]
print "3", D[D_max[3]]
print ""

print "Minimums:", D_min
print "0", D[D_min[0]]
print "1", D[D_min[1]]
print "2", D[D_min[2]]
print "3", D[D_min[3]]
print ""

# Obtain the evidence for all datasets and models
sum = np.sum(evidence, axis=1)
print "Sum:", sum
#index = create_index_set(np.sum(evidence,axis=0))
index = create_index_set(evidence)

# PLOTS #
# LaTeX
pb.rc('text', usetex=True)
pb.rc('font', family='serif')
pb.plot(evidence[0,index],'m', label= r"\Pr($\mathcal{D}$|$M_0$)")
pb.plot(evidence[1,index],'b', label= r"\Pr($\mathcal{D}$|$M_1$)")
pb.plot(evidence[2,index],'r', label= r"\Pr($\mathcal{D}$|$M_2$)")
pb.plot(evidence[3,index],'g', label= r"\Pr($\mathcal{D}$|$M_3$)")
pb.xlabel(r'Datasets $\mathcal{D}$',fontsize=16)
pb.ylabel(r'\Pr($\mathcal{D}$|$M_i)$',fontsize=16)
pb.title(r'Evidence of Datasets given some Models',fontsize=18)
pb.legend()
pb.show()

# Obtain the evidence of the dataset d, for all models and specific theta
#E = evidence_models_theta(d, theta)
#for i in range(len(E)):
#	print "P(D|H_" + str(i) + ", theta) = " + str(E[i])
#print ""