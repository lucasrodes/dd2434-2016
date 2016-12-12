# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Number of data points
N = 1000.0
# Real mean
mu = 0.0
tau = 0.8
# Generation of the data
sigma = np.sqrt(1/tau)
X = np.random.normal(mu,sigma,N)

# Sample mean
smean_x = np.mean(X)
smean_x2 = np.mean(X**2)

# Number of iterations of VI iterative algorithm
ITERATIONS = 5

# PRIORS
# mu ~ N(mu| 0, inf)
# tau ~ Gam(tau | 0, 0)

# 1. Initial guess on the mean of tau
mean_tau = 0.5

for i in range(ITERATIONS):
	# 2. Obtain estimation of q_mu(mu)
	mu_N = smean_x
	lambda_N = N*mean_tau
	# Compute mean(mu) and mean(mu^2)
	mean_mu = smean_x
	mean_mu2 = smean_x**2 + 1/(N*mean_tau)
	# 3. Obtain estimation of q_tau(tau)
	a_N = (N+1)/2
	b_N = N/2*(smean_x2-2*smean_x*mean_mu+mean_mu2)
	# Compute mean(tau)
	mean_tau = a_N/b_N

	#print "MU: ("+str(mu_N)+","+str(1/lambda_N)+")"
	#print "TAU: ("+str(a_N)+","+str(b_N)+")"
#mean_tau = 1/np.mean((X-smean_x)**2)
#mean_tau = 1/smean_x2 - smean_x**2
# 0. Initialize mean(tau)
# 1. Obtain mu_N and lambda_N
# 2. Obtain a_N and b_N
