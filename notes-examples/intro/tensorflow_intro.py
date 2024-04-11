from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow import layers
import matplotlib.pyplot as plt
import numpy as np

tf.random.set_seed(1)
print(tf.__version__)

X = np.array([[0,0], [0,1], [1,0], [1,1]]) #x1, x2
print(X)

Y = np.array([[0], [1], [1], [0]]) #XOR gate output
print(Y)

def build_model():
  """
  Builds a simple neural network with 2 input nodes, 2 hidden nodes and 1 output node.
  """
  #build the model, define the number of layers, the number of nodes in each layer, the activation function, and the input shape
  model = keras.Sequential([
    layers.Dense(2, activation='relu', input_shape=[X.shape[1]]),
    layers.Dense(1)
  ])

  optimizer = keras.optimizers.RMSprop(0.01) #optimisation algorithm, Root Mean Square Propagation

  #compile the model
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse']) 
  
  return model

model = build_model()
model.summary()

num_epochs = 300

model.fit(X, Y, epochs=num_epochs, verbose=1)

plt.plot(model.history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

model.predict(X)