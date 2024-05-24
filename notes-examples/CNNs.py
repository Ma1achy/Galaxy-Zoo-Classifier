# CNN using keras.

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# download mnist dataset
(train_ds, validation_ds, test_ds), metadata = tfds.load(
    
'mnist',
split=['train[:80%]', 'train[80%:100%]', 'test'],
with_info=True,
as_supervised=True
)

ntrain = len([image[0] for image in train_ds])
nvalid = len([image[0] for image in validation_ds])
ntest = len([image[0] for image in test_ds])

print('Train sample: ', ntrain)
print('Validation sample: ', nvalid)
print('Test sample: ', ntest)

# take 1 image from training dataset and convert to float64 and normalise it.

data1 = train_ds.map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float64), label)
).take(1)

# get first image data and label and plot it
features, labels = iter(data1).next()
print('image 1 shape: ', np.shape(features))
plt.imshow(features[:,:,0], cmap='gray_r')
plt.show()

# ======================================== Edge Detection Example with convolution ========================================

# horizontal edge detection kernal = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
# vertical edge detection kernal = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]

hor_kernal = [[1,1,1],
              [0,0,0],
              [-1,-1,-1]]

#2D convolution 
output = tf.nn.conv2d(
      input=np.reshape(features, [1,28,28,1]), # batch, height, width, depth
      filters=np.reshape(hor_kernal, [3,3,1,1]), # height, width, in_channels, out_channels
      strides=[1,1,1,1], # amount to move the kernel across the input tensor
      padding="VALID" 
)

print('image shape: ', np.shape(output))
plt.imshow(output[0,:,:,0], cmap='gray')
plt.show()

# NOTE: output image shape is 26x26 because of padding="VALID" and 28x28 input image size. (valid padding means no padding)
# If padding="SAME" then output image size will be same as input image size.

vert_kernal = [[1,0,-1],
              [1,0,-1],
              [1,0,-1]]

output = tf.nn.conv2d(
      input=np.reshape(features, [1,28,28,1]),
      filters=np.reshape(vert_kernal, [3,3,1,1]),
      strides=[1,1,1,1], #in batch, x, y, channel
      padding="VALID"   
)

print('image shape: ', np.shape(output))
plt.imshow(output[0,:,:,0], cmap='gray')
plt.show()

# ======================================== Pooling Example with max pooling ========================================

# The pooling layer is used to reduce the spatial dimensions of the input volume. The max pooling operation 
# returns the maximum value from the portion of the image covered by the kernel.
# Lets plot a 3x3 max pooling output of our image.

output = tf.nn.max_pool(
      input=np.reshape(features, [1,28,28,1]),
      ksize=3,
      strides=1,
      padding="VALID"    
)

print('image shape: ', np.shape(output))
plt.imshow(output[0,:,:,0], cmap='gray')
plt.show()

# if we increase the kernal size to 10, we get even more shrinkage in the output image size.

output = tf.nn.max_pool(
    
      input=np.reshape(features, [1,28,28,1]),
      ksize=10,
      strides=1,
      padding="VALID"   
)

print('image shape: ', np.shape(output))
plt.imshow(output[0,:,:,0], cmap='gray')

# ======================================== Fully connected layer example ========================================

# set a weight matrix of 1 in the 10th and 11th columns and 0 elsewhere.

weights = np.zeros([19,19])
weights[0:18,9]=1
weights[0:18,10]=1

plt.imshow(weights)
plt.show()

output2 = tf.nn.conv2d(
      input=np.reshape(output, [1,19,19,1]), #batch, height, width, depth
      filters=np.reshape(weights, [19,19,1,1]), #height, width, in_channels, out_channels
      strides=[1,1,1,1], 
      padding="VALID"
)

tf.print(output2)

# ======================================== CNN model example ========================================

# First we will take make a training dataset which the model sees and uses to update model parameters,
# a validation dataset that is not used to update model parameters, but ensures that the model is not overfitting,
# and decide when to stop training, and a test dataset that is used to evaluate the model's performance. These
# are sampled from the original dataset.

# we apply augmentation to the training dataset to increase the number of training samples.

# NOTE : we use shuffle to shuffle buffer 100 images batch split into batches of 24 take 1 batch and repeat the process.
# NOTE: order of operations is important. If we take a batch of 24 images and then shuffle, we will shuffle the 24 images

train = train_ds.map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float64), label) 
).shuffle(100
).batch(24)

valid = validation_ds.map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float64), label) 
).shuffle(100
).batch(24)

test = test_ds.map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float64), label) 
).batch(100)

# We can now build a CNN model. Let's start with a simple model with 2 convolutional layers each with 10 3x3 kernals,
# 2 max pooling layers with 3x3 kernal, and 2 fully connected layers.

tf.keras.backend.clear_session() #Clear keras session

num_classes = 10 
input_shape = [28, 28, 1] 

# define the model
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu', name='conv1',input_shape=input_shape), 
  tf.keras.layers.MaxPool2D(pool_size=(3,3), name='pool1'), 
  tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3),strides=1,padding='valid',activation='relu', name='conv2'), 
  tf.keras.layers.MaxPool2D(pool_size=(3,3), name='pool2'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(20, activation='relu',name='dense1'),
  tf.keras.layers.Dense(num_classes, activation='softmax', name='dense2') 
])

model.summary() 

LR = 0.001 # learning rate

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(LR),
    metrics=['accuracy']
)

history = model.fit(train, epochs=500, validation_data=valid)
score = model.evaluate(test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

for image, label in test:
    pred = model.predict(image)
    for idx in label:
        tf.print('predicted:', np.argmax(pred[idx]), '- truth:', label[idx])
    
