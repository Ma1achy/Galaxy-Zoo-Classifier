#The logistic model is given by:
#yhat = σ (x^T w)
#where σ is the sigmoid function. This is an example of a binary classifier.
#logistic regression is used to estimate the probability that an example belongs to a specific binary class.
#yhat = P(Y=1|x)
#There is no closed form solution for the weights, so we use a iterative procedure to find the weights.
#since the activation function is differentiable, we can use gradient descent, and if the loss function is
#convex it is guaranteed to converge to the global minimum.
#gradient descent proposes a new set of weights:
#w -> w - α∇L(w)
#where α is the learning rate, a positive scalar that determines the size of the update. There are several
#ways to choose α. For now we will set it to a small constant value.
#we will attempt to classify the AND and XOR functions but this time use the BCE loss function.
#In the logistic model, the gradient of the BCE loss with respect to the weights can be shown to be:
#∇L(w) = 1/N * X^T (Yhat - Y)
#gradient descent for logistic regression strongly resembles that of the perceptron.

import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(x): 
    return 1/(1 + np.exp(-x))

X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]]) #x1, x2
print(X)

Y = np.array([[0], [0], [0], [1]]) #AND gate output
print(Y)

np.random.seed(0) 
w = 0.1 * np.random.random(size=(3, 1)) #initialise the weights to small random values
print(w)

num_epochs = 10000 
learning_rate = 0.1

for i in range(num_epochs): #gradient descent
    Yhat = sigmoid(np.matmul(X, w)) 
    w -= learning_rate * np.matmul(X.T, Yhat - Y)
print(w)

Yhat = sigmoid(np.matmul(X, w)) 
print(Yhat)

id0 = np.where(Y[:, 0] == 0) 
id1 = np.where(Y[:, 0] == 1)

xx, yy = np.mgrid[-1:2:.01, -1:2:.01] 
Yhat = sigmoid(w[0] + w[1] * xx + w[2] * yy)

plt.figure() 
plt.contourf(xx, yy, Yhat, alpha=0.5) 
plt.scatter(X[id0, 1], X[id0, 2], color='blue') 
plt.scatter(X[id1, 1], X[id1, 2], color='red') 
plt.xlabel('x1', fontsize=16) 
plt.ylabel('x2', fontsize=16) 
plt.colorbar() 
plt.show()

#XOR function:
Y = np.array([[0], [1], [1], [0]]) #XOR gate output
print(Y)

np.random.seed(0)
w = 0.1 * np.random.random(size=(3, 1)) #initialise the weights to small random values
print(w)

for i in range(num_epochs): #gradient descent
    Yhat = sigmoid(np.matmul(X, w)) 
    w -= learning_rate * np.matmul(X.T, Yhat - Y)
print(w)

Yhat = sigmoid(np.matmul(X, w)) 
print(Yhat)

id0 = np.where(Y[:, 0] == 0) 
id1 = np.where(Y[:, 0] == 1)
xx, yy = np.mgrid[-1:2:.01, -1:2:.01] 
Yhat = sigmoid(w[0] + w[1] * xx + w[2] * yy)

plt.figure() 
plt.contourf(xx, yy, Yhat, alpha=0.5) 
plt.scatter(X[id0, 1], X[id0, 2], color='blue') 
plt.scatter(X[id1, 1], X[id1, 2], color='red') 
plt.xlabel('x1', fontsize=16) 
plt.ylabel('x2', fontsize=16) 
plt.colorbar() 
plt.show()

#logistic regression correctly predicts the AND function, but fails to predict the XOR function, giving all examples a probability of 1/2.
#Similar to the peceptron, it is not able to classify all examples correctly if the training data is not linearly separable.