import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
import sys

# Generate data
mu = 0
tau = 1
N = 100
x = np.random.normal(loc=mu, scale=tau**(-0.5), size=N)

# Define range of mu and tau + function to plot normal_gamma distribution
# in this grid
range_mu = np.arange(-1,1,.025)
range_tau = np.arange(0,2,0.025)


def normal_gamma(mean, precision, shape, rate):
	normal = stats.norm(loc=mean, scale=(precision*range_tau)**(-.5))
	gamma = stats.gamma(a=shape, scale=1.0/rate)
	return [[normal.pdf(u)[t]*gamma.pdf(range_tau[t]) for u in range_mu] for t in range(len(range_tau))]

p = normal_gamma(0,10,5,6)
plt.contour(range_mu,range_tau,p,colors='red')
plt.show()