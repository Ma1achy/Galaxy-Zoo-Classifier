import numpy as np
import matplotlib.pyplot as plt

#AND function:
X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]]) #x1, x2
print(X)

inv = np.linalg.inv(np.matmul(X.T, X))
Y = np.array([[0], [0], [0], [1]]) #AND gate output
print(Y)

plt.scatter(X[:,1], X[:,2],c=np.reshape(Y,4), edgecolors='black', cmap='gray')
plt.show() #AND gate outputs

w = np.matmul(inv,  np.matmul(X.T, Y)) #weights, found by choosing to model as a linear regression problem -> so MSE loss function is used.
print(w) 

Yhat = np.matmul(X, w)
print(Yhat)

plt.scatter(X[:,1], X[:,2],c=np.reshape(Yhat,4), edgecolors='black', cmap='gray') #plot the predictions
plt.show() #AND gate predictions

#if we have new points, we cab use the trained model to predict the Y values

n = 100
Xtest = np.transpose(np.reshape(np.append(np.ones(n),np.random.uniform(0,1,n*2)),(3,n)))
Yhattest = np.matmul(Xtest, w)

plt.scatter(Xtest[:,1], Xtest[:,2],c=np.reshape(Yhattest,n), edgecolors='black', cmap='gray')
plt.xlim([0,1])
plt.ylim([0,1])
plt.show() 

#XOR function:
Y = np.array([[0], [1], [1], [0]])
print(Y)

w = np.matmul(inv,  np.matmul(X.T, Y))
print(w)

Yhat = np.matmul(X, w)
print(Yhat)

plt.scatter(X[:,1], X[:,2],c=np.reshape(Yhat,4), edgecolors='black', cmap='gray')
plt.show() #XOR gate outputs

#The model is not able to predict the XOR gate outputs, outputting 1/2 for all examples.

n=100
Xtest = np.transpose(np.reshape(np.append(np.ones(n),np.random.uniform(0,1,n*2)),(3,n)))
Yhattest = np.matmul(Xtest, w)

plt.scatter(Xtest[:,1], Xtest[:,2],c=np.reshape(Yhattest,n), edgecolors='black', cmap='gray')
plt.xlim([0,1])
plt.ylim([0,1])
plt.show() 

#This is true for the test samples as well.