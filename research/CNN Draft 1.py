#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:02:19 2024

@author: aroushijimulia
"""

"""
Convolusional Neural Network 
Draft 1 - Started 04/03/24
"""

#Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.model_selection import train_test_split

#################################### Functions ########################################

def load_class_data(file_path):
    """
    Summary: Loads class label data in one-hot format
    Input: Path to parquet file
    Output: Pandas dataframe with one hot class labels
    """
    df = pd.read_parquet(file_path)
    class_labels = df['label'] + 1
    class_labels_one_hot = to_categorical(class_labels, num_classes=8) #turns into one-hot
    return class_labels_one_hot


def load_image(full_image_path):
    """
    Summary: Loads and resizes image
    Input: Path to image file
    Output: Image tensor
    """
    image = tf.io.read_file(full_image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  
    image = image / 255.0  
    return image  

def load_images(catalog, image_path):
    """
    Summary: Loads image tensors into a list
    Input: Main pandas dataframe which includes image locations, image paths
    Output: List of image tensors
    """
    images = []
    for relative_path in catalog['file_loc']:
        relative_path = relative_path[-29:] #Picks out relevant bit of full path
        full_image_path = os.path.join(image_path, str(relative_path))
        image = load_image(full_image_path)  
        images.append(image)
    return images

def display_image_from_catalog(catalog, image_path, index):
    """
    Summary: Can be used to open individual images if needed
    Input: Main pandas dataframe, image path, image index
    Output: Displays image
    """
    full_image_path = os.path.join(image_path, catalog['file_loc'].iloc[index])
    im = Image.open(full_image_path)
    im.show()

####################################### File Paths ###########################################

"""
At the moment, these are specific to my Mac but can be changed to be used with Malachy's computer'
"""
file_path = r'/Users/aroushijimulia/Downloads/GitHub/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset/gz2_train_catalog.parquet'
image_path = r'/Users/aroushijimulia/Downloads/GitHub/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset/images'

####################################### Analysis #############################################

#Loads class labels into class_data
class_data = load_class_data(file_path)

#Loads all parquet data into catalog
catalog_main = pd.read_parquet(file_path)  

#Keep specified columns so code runs faster
columns_to_keep = ['smooth-or-featured-gz2_smooth_fraction', 'smooth-or-featured-gz2_featured-or-disk_fraction', 'smooth-or-featured-gz2_artifact_fraction', 'file_loc', 'subfolder', 'filename']
catalog_1 = catalog_main[columns_to_keep]

for i in range(0,8):
    column_name = i
    catalog_1[str(column_name)] = class_data[:,i]
    
# Extracting target columns directly without one-hot encoding
target_columns = ['smooth-or-featured-gz2_smooth_fraction', 'smooth-or-featured-gz2_featured-or-disk_fraction', 'smooth-or-featured-gz2_artifact_fraction']
y = catalog_1[target_columns].values
   
catalog_1['file_loc'] = catalog_1['file_loc'].str[-29:]

batch_size = 32
original_size = 424
target_size = 224
 
#%%

############################# Define ResNet 50 model architecture ############################

def build_model():
    resnet_model = Sequential()

    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                       input_shape=(224,224,3),
                       pooling='avg',
                       weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable = True

    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(3, activation='softmax')) 
    
    # Compile the model with Mean Squared Error Loss for a regression task
    resnet_model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy']) 
        
    return resnet_model

model = build_model()

#%%
# Splitting the data into train+val and test sets first
train_val_catalog, test_catalog = train_test_split(catalog_1, test_size=0.1, random_state=42)

# Further splitting the train+val into actual train and val sets
train_catalog, val_catalog = train_test_split(train_val_catalog, test_size=0.2, random_state=42)

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

train_iterator = train_datagen.flow_from_dataframe(
    dataframe=train_catalog,
    directory=image_path,  # Assuming file_loc is relative to this directory
    x_col='file_loc',
    y_col=target_columns,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='raw'  # 'raw' because your labels are continuous values
)

# Validation data generator (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

val_iterator = val_datagen.flow_from_dataframe(
    dataframe=val_catalog,
    directory=image_path,
    x_col='file_loc',
    y_col=target_columns,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='raw'
)

history = model.fit(
    train_iterator,
    steps_per_epoch=len(train_catalog) // batch_size,
    validation_data=val_iterator,
    validation_steps=len(val_catalog) // batch_size,
    epochs=10  # Set the number of epochs according to your experiment
)



