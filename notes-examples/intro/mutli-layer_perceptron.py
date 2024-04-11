#multi-layer perceptron model used to reproduce the XOR function is given by:
#Yhat = (ReLU(XW))w,
#with one hidden layer containing two hidden units.
#the optimal weights are W =[[0, -1], [1, 1], [1, 1]] and w = [[0], [1], [2]].

import numpy as np 
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]]) 
X = np.hstack((np.ones(shape=(X.shape[0], 1)), X)) 
print(X)

Y = np.array([[0], [1], [1], [0]]) 
print(Y)

W = np.array([[0, -1], [1,1], [1,1]], dtype=float) 
print(W)

w = np.array([[0], [1], [-2]], dtype=float) 
print(w)

h = np.maximum(np.matmul(X, W), 0) 
h = np.hstack((np.ones(shape=(h.shape[0], 1)), h)) 
print(h)

Yhat = np.matmul(h, w) 
print(Yhat)

#It is instructive to see how the MLP is able to learn the XOR function
#by plotting the rep- resentation of data it learns in the hidden space
id0 = np.where(Y[:, 0] == 0) 
id1 = np.where(Y[:, 0] == 1)

plt.figure() 
plt.scatter(X[id0, 1], X[id0, 2], color='blue') 
plt.scatter(X[id1, 1], X[id1, 2], color='red') 
plt.xlabel('x1', fontsize=16) 
plt.ylabel('x2', fontsize=16) 
plt.show()

plt.figure() 
plt.scatter(h[id0, 1], h[id0, 2], color='blue') 
plt.scatter(h[id1, 1], h[id1, 2], color='red') 
plt.xlabel('h1', fontsize=16) 
plt.ylabel('h2', fontsize=16) 
plt.show()

#The original data is not linearly separable, but in the representation space
#it has mapped the two points with output 1 into a single point, and this space is now separable