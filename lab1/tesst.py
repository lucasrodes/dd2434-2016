import matplotlib.pyplot as pb
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from math import pi
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

## GENERATE SYNTHETIC DATA ##
# Input
x = np.arange(-1,1,0.01)
x = np.vstack([x,np.ones(len(x))])
# Weights
W = np.array([-1.3,0.5])
# Gaussian noise
sigma = np.sqrt(3)
epsilon = np.random.normal(0,sigma,np.size(x,1))
# Output
Y = np.dot(W,x)+epsilon


L = 400
mu = np.array([[0],[0]])
sigma = 2*np.eye(2)

def normal_pdf(x):
	return 1/(2*pi*det(sigma))*np.exp(-0.5*np.dot(np.dot(np.transpose(mu-x),inv(sigma)),(mu-x)))

def prior(X, Y):
	Nx = np.size(X,1)
	Ny = np.size(Y,1)
	Z = np.zeros((Nx,Ny))
	for i in range(Nx):
   		for j in range(Ny):
   			Z[i,j] = normal_pdf(np.array([[X[i,j]],[Y[i,j]]]))
	return Z

w0 = np.linspace(-2,2,L)
w1 = np.linspace(-2,2,L)
W0, W1 = np.meshgrid(w0,w1)

pb.pcolor(W0, W1, prior(W0,W1))
pb.xlabel('w_0')
pb.ylabel('w_1')
pb.title('Prior P(W)')
pb.show()
