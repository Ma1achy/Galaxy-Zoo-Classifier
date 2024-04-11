import numpy as np 
import matplotlib.pyplot as plt 

#perceptron model is given by:
#yhat = ϴ (x^T w)
#where ϴ is the Heaviside step function. This is an example of a binary classifier.
#due to the non-linearty of the Heaviside step function we cannot solve for the weights using the normal equations
#we can use the perceptron learning algorithm to iteratively find the weights depending on the size of the prediction error.
#1/ Initialise the weights to 0 or small random values
#2/ Iteravitely update the weights by w -> w + αX^T (Yhat - Y), where α is the learning rate.
#3/ Repeat step 2 until a stopping codition is met, this could be a maximum number of iterations or that the 
# loss has not improved significantly from the previous iteration.

#AND function:

X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]]) #x1, x2
print(X)

Y = np.array([[0], [0], [0], [1]]) #AND gate output
print(Y)

np.random.seed(0) 
w = 0.1 * np.random.random(size=(3, 1)) #initialise the weights to small random values
print(w)

num_epochs = 10 
learning_rate = 0.1 #α

for i in range(num_epochs): 
    Yhat = np.heaviside(np.matmul(X, w), 0) 
    w -= learning_rate * np.matmul(X.T, Yhat - Y)
print(w)

#check the classification of the training samples
Yhat = np.heaviside(np.matmul(X, w), 0) 
print(Yhat)

#plot the descision boundary for all values of x
id0 = np.where(Y[:, 0] == 0) 
id1 = np.where(Y[:, 0] == 1)

xx, yy = np.mgrid[-1:2:.01, -1:2:.01] 
Yhat = np.heaviside(w[0] + w[1] * xx + w[2] * yy, 0)

plt.figure() 
plt.contourf(xx, yy, Yhat, alpha=0.5) 
plt.scatter(X[id0, 1], X[id0, 2], color='blue') 
plt.scatter(X[id1, 1], X[id1, 2], color='red') 
plt.xlabel('x1', fontsize=16) 
plt.ylabel('x2', fontsize=16) 
plt.show()

#XOR function:
Y = np.array([[0], [1], [1], [0]]) #XOR gate output
print(Y)

np.random.seed(0) 
w = 0.1 * np.random.random(size=(3, 1)) #initialise the weights to small random values
print(w)

for i in range(num_epochs): 
    Yhat = np.heaviside(np.matmul(X, w), 0) 
    w -= learning_rate * np.matmul(X.T, Yhat - Y)
print(w)

#check the classification of the training samples
Yhat = np.heaviside(np.matmul(X, w), 0) 
print(Yhat)

#plot the descision boundary for all values of x
id0 = np.where(Y[:, 0] == 0) 
id1 = np.where(Y[:, 0] == 1)

xx, yy = np.mgrid[-1:2:.01, -1:2:.01] 
Yhat = np.heaviside(w[0] + w[1] * xx + w[2] * yy, 0)

plt.figure() 
plt.contourf(xx, yy, Yhat, alpha=0.5) 
plt.scatter(X[id0, 1], X[id0, 2], color='blue') 
plt.scatter(X[id1, 1], X[id1, 2], color='red') 
plt.xlabel('x1', fontsize=16) 
plt.ylabel('x2', fontsize=16) 
plt.show()

#The perceptron manages to classify the AND gate outputs correctly, but not the XOR gate outputs, giving all examples the same value
#or class. This is because the perceptron is a linear classifier and will never be able to classify all examples correctly if the training
#data is not linearly separable, that is all examples of one class cannot be seperated from the other by a hyperplane.
#This is the case for the XOR gate outputs.

#The perceptron algorith is guaranteed to converge on some solution if the training data is linearly separable, but it may still pick any
#solution, depending on the inital weights and the training procedure. In the case of the AND example, there are infinitely many solutions
#seperating the two classes.