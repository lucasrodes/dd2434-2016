import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,4)
y = np.linspace(0,1)

def f(x, y):
    return y * np.sin(x) 

X, Y = np.meshgrid(x,y)
Z = np.zeros((50,50))

for i in range(50):
   for j in range(50):
       Z[i,j] = f(X[i,j],Y[i,j])

plt.pcolor(X, Y, Z)
plt.show()
