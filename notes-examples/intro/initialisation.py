# during backpropagation, it is important that the gradients do not become too large or too small, as
# the alogirthm progresses down to the lower layers of the network. This is known as the vanishing gradients
# or exploding gradients problem. This can be mitigated by using activation functions that do not saturate 
# too quickly, and by using batch normalisation.

# consider a MLP with 100 hidden layers, each with 100 hidde units (with zero bias) and a sigmoid activation function.
# we assume the input data (with dimension 100) and the weights are normally distributed with mean 0 and variance 1.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def sigmoid(x):
  return 1/(1 + np.exp(-x))

hidden_units = 100

X = np.random.normal(size=(1000, hidden_units))

def test_initialisation(X, num_layers, activation, scale): 
    """
    Test the initialisation of the weights and biases for a MLP with 100 hidden layers, 
    each with 100 hidden units (with zero bias) and a sigmoid activation function.
    """
    a = np.copy(X)
    plt.figure(figsize=(15,10))
  
    for i in range(num_layers):
        
        W = np.random.normal(scale=scale, size=(hidden_units, hidden_units)) #initialise the weights to small random values
        z = np.matmul(a, W)
        a = np.copy(activation(z))

        if i % 20 == 0: #plot the activation and its derivative every 20 layers
            
            ax = plt.subplot(2,2,1)
            _ = ax.hist(a.flatten(), label='Layer ' + str(i))
            ax.set_xlabel('Φ', fontsize=14)
            ax = plt.subplot(2,2,2)
            _ = ax.hist(a.flatten() * (1 - a.flatten() ), label='Layer ' + str(i))
            ax.set_xlabel("Φ'", fontsize=14)
            
    plt.legend()
    plt.show()
        
test_initialisation(X, 100, sigmoid, 1)

# ideally, we preserve the signal variance as we pass through the network, such that it neither vanishes nor saturates.
# The variance of the output of a neuron should therefore be equal to the variance of its input. 
# In order for the variance ot to be preserved, we need to set the variance of the weights to 1/number_of_inputs.
# During backpropagationm, we would like to preserve the variance of the gradients as well, which can be achieved by setting 
# the variance of the weights to 1/number_of_outputs, where number_of_outputs is the number outputs from a neuron.

# a heuristic for balancing both requirements is a weight initialsation of either:
# W = uniform(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))
#This is Xavier initialisation.

#if the weights are drawn from a uniform distribution and the inital weights are drawn from a normal distribution.
# W = normal(0, sqrt(2/(n_inputs + n_outputs)))
#This is Glorot initialisation.

test_initialisation(X, 100, sigmoid, np.sqrt(1 / hidden_units))