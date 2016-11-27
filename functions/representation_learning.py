# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import pylab as pb
from matplotlib import rc
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import random


import  numpy  as np
import  scipy  as sp
import  scipy.optimize  as opt


# For notation issues take a look at the report #

def f(W):
	# Return the -log type-II Likelihood of W #
	# 			   We assume mu = 0    	      #

	# Convert vector to matrix
	W = np.reshape(W,(10,2))
	# Obtain the covariance of the marginal likelihood
	C_w = 1./beta_est**.5*np.eye(10) + np.dot(W,np.transpose(W))
	C_w = 1./beta_est**.5*np.eye(10) + np.dot(W,np.transpose(W))
	A = N*D/2.0*np.log(2*pi)
	B = N/2.0*np.log(det(C_w))
	C = 1/2.0*np.trace(np.dot(inv(C_w),np.dot(np.transpose(Y),Y)))
	return A+B+C

def dfx(W):
	# Return the derivative of -log type-II Likelihood of W #
	# 					We assume mu = 0 			        #
	W = np.reshape(W,(10,2))
		# Obtain the covariance of the marginal likelihood
	C_w = 1./beta_est**.5*np.eye(10) + np.dot(W,np.transpose(W))
	# Define the gradient and obtain its coefficients
	gradient = np.empty(W.shape)
	for i in range(gradient.shape[0]):
		for j in range(gradient.shape[1]):
			# Obtain partial derivative of dWW^T/dW_ij J = np.zeros(np.shape(W))
			J = np.zeros(np.shape(W))
			J[i,j] = 1
			dWW_ij = np.dot(J,np.transpose(W)) + np.dot(W,np.transpose(J))
			dB = N/2.0 * np.trace(np.dot(inv(C_w),dWW_ij))
			dC = -1/2.0 * np.trace( np.dot(np.transpose(Y), np.dot(Y, np.dot( inv(C_w), np.dot( dWW_ij, inv(C_w) ) )  )  )  )
			gradient[i,j]= dB+dC
	gradient = np.reshape(gradient,(20,))
	return  gradient


## GENERATE THE DATA ##

# UNSEEN TO US
# Input data
N = 100
X = np.linspace(1,4*pi, N) # 1x100
# Apply non linear function
f_nl = np.transpose(np.array([X*np.sin(X), X*np.cos(X)])) # 100x2
# Matrix A
A = np.random.normal(0,1,[10,2]) #10x2
# Noise
f_l = f_nl*np.matrix(np.transpose(A)) # 100 x 10
beta = 1
noise = np.random.multivariate_normal(np.zeros(10), 1./beta**(0.5)*np.eye(10), N)

# OBSERVABLE
# Apply linear function
Y = f_l + noise
D = np.size(Y,1)

# INFER INPUT DATA FROM OUTPUT
# Estimate of the noise precision
beta_est = 1
# Initial estimate of W
W_init = 20*np.random.randn(20)
W_init = np.reshape(W_init, (20,))
W_est = 0.2*opt.fmin_cg(f, W_init, fprime=dfx) # Had to scale, some bug to be fixed
W_est = np.reshape(W_est,(10,2))

X_est = np.dot ( Y, np.dot ( W_est, inv(np.dot(np.transpose(W_est), W_est) ) ) )
real = pb.scatter(f_nl[:,0],f_nl[:,1],color='blue', label='Real data')
inf = pb.scatter(X_est[:,0],X_est[:,1], color='red', label='Infered data')
pb.xlabel(r'$y_1$',fontsize=16)
pb.ylabel(r'$y_2$',fontsize=16)
pb.legend([real, inf], [r'Real data', r'Infered data'])
pb.title(r'Comparison of real data and infered')
pb.show()

real = pb.scatter(f_nl[:,0],f_nl[:,1],color='blue', label='Real data')
pb.xlabel(r'$y_1$',fontsize=16)
pb.ylabel(r'$y_2$',fontsize=16)
pb.title(r'Real data, i.e. $f_{lin}(x)$')
pb.show()


