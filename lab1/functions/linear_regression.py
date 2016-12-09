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

# To sample from a multivariate Gaussian
# f = np.random.multivariate_normal(mu,K)
# To compute a distance matrix between two sets of vectors
# D = cdist(x1,x2)
# To compute the exponential of all elements in a matrix
# E = np.exp(D)


# FUNCTIONS #
def normal_pdf_prob(x, mean, cov):
	return multivariate_normal.pdf(x, mean, cov)


# PRIOR
def normal_prior_pdf(mean, cov):
	Nx = np.size(W0,1)
	Ny = np.size(W1,1)
	Z = np.zeros((Nx,Ny))
	for i in range(Nx):
   		for j in range(Ny):
   			Z[i,j] = normal_pdf_prob(np.array([W0[i,j],W1[i,j]]), mean, cov)
	return Z


# LIKELIHOOD
def normal_likelihood_pdf(x, y, cov):
	Nx = np.size(W0,1)
	Ny = np.size(W1,1)
	Z = np.zeros((Nx,Ny))
	for i in range(Nx):
   		for j in range(Ny):
   			W = np.array([W0[i,j],W1[i,j]])
   			mean = np.inner(W,x)
   			Z[i,j] = normal_pdf_prob(y, mean, cov)
	return Z


def plot_normal_pdf_sample_space(mean, cov, nsamples, name=None):
	F = np.random.multivariate_normal(np.transpose(mean)[0],cov,nsamples)
	for f in F:
		pb.plot(f)
	pb.show()


# GENERATE SYNTHETIC DATA #
# Input
X = np.linspace(-1,1,201)
X = np.vstack([X,np.ones(len(X))])
X = np.matrix(np.transpose(X))
# Weights
W = np.array([-1.3,0.5])
print "Real W = ", W
# Gaussian noise
beta= 1./.3
epsilon = np.random.normal(0,beta**(-.5),np.size(X,0))
# Output
Y = np.transpose(W*np.transpose(X)+epsilon)

# GLOBAL PARAMETERS, INITIALIZATION #
L = 150
w0 = np.linspace(-2,2,L)
w1 = np.linspace(-2,2,L)
W0, W1 = np.meshgrid(w0,w1)
alpha = 2
mean_prior = np.array([0,0])
cov_prior = alpha*np.identity(2)
cov_like = beta**(-1)

# LaTeX
pb.rc('text', usetex=True)
pb.rc('font', family='serif')

# PRIOR #
prior = normal_prior_pdf(mean_prior,cov_prior)
pb.subplot(4,3,2)
pb.pcolor(W0, W1, prior)
pb.xlabel(r'$w_0$', fontsize=16)
pb.ylabel(r'$w_1$', fontsize=16)
pb.title(r'Prior/Posterior', fontsize=17)


pb.subplot(4,3,1)
pb.title(r'Likelihood', fontsize=17)
pb.subplot(4,3,3)
nsamples = 20
F = np.random.multivariate_normal(mean_prior,cov_prior,nsamples)
for f in F:
	pb.plot(f)
pb.title(r'Data space', fontsize=17)

#plot_normal_pdf_sample_space(mean_prior, cov_prior, 6, name=r'prior_samples')

# N OBSERVATIONS #
N = 20
I = np.random.randint(0, 201, N)
posterior = prior
cont = 0
for i in I:
	x_i = X[i]
	y_i = Y[i]
	print "Observed: [",x_i[0,0],",",y_i[0,0],"]"

	# LIKELIHOOD #
	likelihood = normal_likelihood_pdf(x_i, y_i, cov_like)

	# POSTERIOR #
	posterior *= likelihood
	posterior /= sum(sum(posterior))
	
	if (cont == 0 or cont == 1 or cont == 19):
		pb.subplot(4,3,4+3*cont%17)
		pb.pcolor(W0, W1, likelihood)
		pb.xlabel(r'$w_0$', fontsize=16)
		pb.ylabel(r'$w_1$', fontsize=16)
		pb.subplot(4,3,5+3*cont%17)
		pb.pcolor(W0, W1, posterior)
		pb.xlabel(r'$w_0$', fontsize=16)
		pb.ylabel(r'$w_1$', fontsize=16)

		X_obs = X[I[:cont+1]]
		Y_obs = Y[I[:cont+1]]
		# Update covariance and mean of posterior
		cov_post = inv(inv(cov_prior)+beta*np.transpose(X_obs)*X_obs)	
		_mean_post = np.dot(inv(cov_prior),mean_prior)+np.transpose(beta*np.transpose(X_obs)*Y_obs)
		mean_post = np.inner(cov_post,_mean_post)
		mean_post = np.squeeze(np.asarray(mean_post))
		# DATA SPACE #
		pb.subplot(4,3,6+3*cont%17)
		F = np.random.multivariate_normal(mean_post,cov_post,nsamples)
		for f in F:
			print f
			y0 = f[0]*(-3)+f[1]
			y1 = f[0]*(3)+f[1]
			pb.plot([-2, 2],[y0, y1])
			pb.axis([-2, 2, -3.3, 3.3])

			for j in I[:cont+1]:
				pb.plot(X[j][0,0], Y[j][0,0], 'ro')
	cont += 1
	#i += 3	




print "mean = ", mean_post
print "cov = ", cov_post

#pb.subplot(4,3,8)
#posterior_th = normal_prior_pdf(mean_post,cov_post)
#pb.pcolor(W0, W1, posterior_th)
pb.savefig("q11.pdf")
pb.show()
