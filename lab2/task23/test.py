# -*- coding: utf-8 -*-
#import numpy as np
#import matplotlib.pyplot as plt
#from math import pi
#mu = 0
#sigma = 1

# DEFINE GRID
#x = np.arange(-10,10,0.1)
#y = np.arange(-10,10,0.1)
#xv, yv = np.meshgrid(x, y)

#f = 1/np.sqrt(2*pi*sigma)*np.exp(-1/(2*sigma**2)*(x - mu)**2)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import pandas as pd

# Generate data
mu = 0
tau = 2
N = 10
x = np.random.normal(loc=mu, scale=tau**(-0.5), size=N)

# Define range of mu and tau + function to plot normal_gamma distribution
# in this grid
range_mu = np.arange(-.4,0.7,0.05)#np.arange(-.1,.1,.005)
range_tau = np.arange(0.5,8,0.1)#np.arange(1.7,2.3,0.005)


def normal_gamma(mean, precision, shape, rate):
	normal = stats.norm(loc=mean, scale=(precision*range_tau)**(-.5))
	gamma = stats.gamma(a=shape, scale=1.0/rate)
	return [[normal.pdf(u)[t]*gamma.pdf(range_tau[t]) 
	for u in range_mu] for t in range(len(range_tau))]


# Sample mean and sample variance
sm = np.mean(x)
sv = np.var(x)

# Prior parameters (initial belief, hyper-parameters)
mu_0 = 1
lambda_0 = 1
a_0 = 0.01
b_0 = 0.01
prior = normal_gamma(mu_0,lambda_0,a_0,b_0)

# Posterior parameters
mu_p = (lambda_0*mu_0+N*sm)/(lambda_0 + N)
lambda_p = lambda_0+N
a_p = a_0 + N/2
b_p = b_0 + 0.5*(N*sv + (lambda_0*N*(sm-mu_0))/(lambda_0+N))
posterior = normal_gamma(mu_p,lambda_p,a_p,b_p)

# Estimate posterior
a_N = a_0
b_N = b_0
ITERATIONS = 10
qposterior= []
for i in range(ITERATIONS):
	mean_tau = a_N/b_N
	# 2. Obtain estimation of q_mu(mu)
	mu_N = (lambda_0*mu_0+N*sm)/(lambda_0+N)
	lambda_N = (N+lambda_0)*mean_tau
	qposterior1 = normal_gamma(mu_N,lambda_N,a_N,b_N)
	# Compute mean(mu) and mean(mu^2)
	mean_mu = mu_N
	mean_mu2 = mu_N**2 + 1/lambda_N
	# 3. Obtain estimation of q_tau(tau)
	a_N = a_0+(N+1)/2
	b_N = b_0+.5*(mean_mu2*(N+lambda_0)-2*mean_mu*(N*sm+lambda_0*mu_0)+lambda_0*mu_0**2+sum(x**2))
	# Compute mean(tau)
	qposterior2 = normal_gamma(mu_N,lambda_N,a_N,b_N)
	if i <1:
		qposterior.append(qposterior1)
		qposterior.append(qposterior2)

qposterior.append(qposterior2)
#qposterior = normal_gamma(mu_N,lambda_N,a_N,b_N)

# PLOT results
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

print "preparing plots..."

fig = plt.figure(1)
ax = []

ax.append(fig.add_subplot(221))
ax.append(fig.add_subplot(222))
ax.append(fig.add_subplot(223))
ax.append(fig.add_subplot(224))

ax[0].contour(range_mu,range_tau,prior,colors='red')
ax[0].contour(range_mu,range_tau,posterior,colors='green')
ax[0].set_xlabel(r"(a)")

ax[1].contour(range_mu,range_tau,qposterior[0],colors='red')
ax[1].contour(range_mu,range_tau,posterior,colors='green')
ax[1].set_xlabel(r"(b)")

ax[2].contour(range_mu,range_tau,qposterior[1],colors='red')
ax[2].contour(range_mu,range_tau,posterior,colors='green')
ax[2].set_xlabel(r"(c)")

ax[3].contour(range_mu,range_tau,qposterior[2],colors='red')
ax[3].contour(range_mu,range_tau,posterior,colors='green')
ax[3].set_xlabel(r"(d)")

fig.text(0.5, 0.04, r'$\mu$', fontsize=17, ha='center')
fig.text(0.04, 0.5, r'$\tau$', fontsize=17, va='center', rotation='vertical')
fig.suptitle(r'Estimation of $p(\mu,\tau|D)$ using Variational Inference algorithm', fontsize=17)

print "plotting..."
plt.show()

real_prior = [mu_0, lambda_0, a_0, b_0]
real_posterior = [mu_p, lambda_p, a_p, b_p]
estim = [mu_N, lambda_N, a_N, b_N]

print "PRIOR = ", real_prior
print "POSTERIOR = ", real_posterior
print "ESTIMATED = ", estim
'''plt.contour(range_mu,range_tau,posterior,colors='green')
plt.contour(range_mu,range_tau,qposterior1,colors='red')
plt.title(r'Posterior $p(\mu,\tau | D)$', fontsize=20)
plt.xlabel(r'$\mu$', fontsize=16)
plt.ylabel(r'$\tau$', fontsize=16)
plt.show()

plt.contour(range_mu,range_tau,posterior,colors='green')
plt.contour(range_mu,range_tau,qposterior2,colors='red')
plt.title(r'Posterior $p(\mu,\tau | D)$', fontsize=20)
plt.xlabel(r'$\mu$', fontsize=16)
plt.ylabel(r'$\tau$', fontsize=16)
plt.show()'''
