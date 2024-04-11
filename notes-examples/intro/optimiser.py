#the rosenbrock function is a non-convex function used for bench-marking optimization methods.
#it is defined as f(x,y) = (1-x)^2 + 100(y-x^2)^2
#and has a global minimum at (x,y) = (1,1)

#benchmark the optimisers included with tensorflow, SGD, SGD with momentum, RMSprop, and Adam
#starting from (x,y) = (0,1).

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

def rosenbrock(x, y): #rosenbrock function
	return (1 - x)**2 + 100 * (y - x**2)**2

#benchmark function
def benchmark(f, opt, iterations, x_init, y_init):
    
  x = tf.Variable(x_init) 
  y = tf.Variable(y_init) 
  history = []
  
  for i in range(iterations):
    with tf.GradientTape() as tape:
      z = f(x, y)
      
    grads = tape.gradient(z, [x, y])
    history.append([x.numpy(), y.numpy(), grads[0].numpy(), grads[1].numpy()])
    
    processed_grads = [g for g in grads]
    grads_and_vars = zip(processed_grads, [x, y])
    
    opt.apply_gradients(grads_and_vars)
    
  return np.array(history)

iterations = 100
x_init = 0.0
y_init = 1.0

#benchmark the optimisers
sgd_history = benchmark(rosenbrock, keras.optimizers.SGD(learning_rate=0.001), iterations, x_init, y_init)
momentum_history = benchmark(rosenbrock, keras.optimizers.SGD(learning_rate=0.001, nesterov=True, momentum=0.2), iterations, x_init, y_init)
rmsprop_history = benchmark(rosenbrock, keras.optimizers.RMSprop(learning_rate=0.01), iterations, x_init, y_init)
adam_history = benchmark(rosenbrock, keras.optimizers.Adam(learning_rate=0.05), iterations, x_init, y_init)
     
xx, yy = np.mgrid[-1:2:.01, -0.5:2:.01]
zz = rosenbrock(xx, yy)

#plotting
plt.figure(figsize=(15, 5))
ax = plt.subplot(1, 2, 1)
ax.contourf(xx, yy, np.log(zz), alpha=0.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(sgd_history[:, 0], sgd_history[:, 1], label='SGD')
ax.plot(momentum_history[:, 0], momentum_history[:, 1], label='SGD + momentum')
ax.plot(rmsprop_history[:, 0], rmsprop_history[:, 1], label='RMSprop')
ax.plot(adam_history[:, 0], adam_history[:, 1], label='Adam')
ax.set_xlim([-1,1.5])
ax.set_ylim([-0.5,2])
ax.legend(loc='upper left')
ax = plt.subplot(1, 2, 2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Distance from true solution')
ax.plot(((sgd_history[:, 0] - 1)**2 + (sgd_history[:, 1] - 1)**2)**0.5, label = 'SGD')
ax.plot(((momentum_history[:, 0] - 1)**2 + (momentum_history[:, 1] - 1)**2)**0.5, label = 'SGD + momentum')
ax.plot(((rmsprop_history[:, 0] - 1)**2 + (rmsprop_history[:, 1] - 1)**2)**0.5, label = 'RMSprop')
ax.plot(((adam_history[:, 0] - 1)**2 + (adam_history[:, 1] - 1)**2)**0.5, label = 'Adam')
ax.legend()
plt.tight_layout()
plt.show()