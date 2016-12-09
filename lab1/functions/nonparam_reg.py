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

def kernel(xi, xj, l):
	return sigma_f*np.exp(-(xi-xj)**2/l**2)

def Kernel(X,Y,l):
	return sigma_f*np.exp(-cdist(X, Y, 'sqeuclidean')/(l*l))

def get_gram_matrix(X, l):
	for i in range(len(X)):
		for j in range(i, len(X)):
			K[i,j] = kernel(X[i],X[j], l)
			K[j,i] = K[i,j]
	return K

# LaTeX
pb.rc('text', usetex=True)
pb.rc('font', family='serif')

q12 = False
if (q12):# Define the prior
	N = 100
	X = np.linspace(-1,1,N)
	K = np.zeros([N,N])
	L = [0.01, 0.1, 1, 10]
	sigma_f = 1

	count = 1
	for l in L:
		K = get_gram_matrix(X, l)
		F = np.random.multivariate_normal(np.zeros(N),K,10)
		pb.subplot(2,2,count)
		for f in F:
			pb.plot(X,f)
		
		pb.xlabel(r'$x$')
		pb.ylabel(r'$f(x)$')
		pb.title('$l ='+str(l)+'$')
		count = count + 1

	pb.show()

def get_posterior(xx, X, Y,l):
    #Xstar = np.array([xStar])
    xx = xx[:, None]
    

    return mu, sigma

# OBSERVED DATA #
X = np.array([-pi, -3*pi/4,-pi/2,0, pi/2, 3*pi/4, pi])
beta= 1./.5
epsilon = np.sqrt(0.5) * np.random.randn(7)
Y = np.sin(X)+epsilon
l = 1

# Fitting 7 observations 
pb.plot(X, Y,'ro')
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
pb.plot(x,np.sin(x), color = 'green')
x = x[:,None]
X = X[:, None]
Y = Y[:, None]
k = Kernel(x,X,l)
C = Kernel(X,X,l) + .3*np.eye(np.size(X,0))
mu = np.dot(np.dot(k,inv(C)),Y)
c = Kernel(x, x,l)
sigma = c- np.dot(np.dot(k,C),np.transpose(k))
mu = np.reshape(mu, (1000,))
pb.plot(x,mu, color = 'blue')
pb.xlabel(r'$x$')
pb.ylabel(r'$y$')
pb.title(r'$l = '+str(l)+'$')
print sigma

# Sampling from the posterior
Post = np.random.multivariate_normal(mu,sigma,20)
pb.figure() # open new plotting window?
pb.plot(X,Y,'ro')
for i in range(20):
    pb.plot(x[:],Post[i,:])
pb.xlabel(r'$x$')
pb.ylabel(r'$y$')
pb.title(r'Sampling from the posterior')
pb.show()

# Prior
#K = get_gram_matrix(X, l)

# Posterior


