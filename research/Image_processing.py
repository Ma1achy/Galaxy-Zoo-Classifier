#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:06:17 2024

@author: aroushijimulia
"""
#Import libraries
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from skimage import io, transform
from skimage.util import random_noise
from scipy import fftpack
import random
from PIL import Image
from skimage import util, filters

#Define file paths
root_dir = r"/Users/aroushijimulia/Downloads/GitHub/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset/images"
parquet_file = r'/Users/aroushijimulia/Downloads/GitHub/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset/gz2_train_catalog.parquet'
aug_images_dir = '/Users/aroushijimulia/Downloads/GitHub/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset/aug_images'

# Load image file locations
catalog = pd.read_parquet(parquet_file)
# Use the first 100 images for testing augmentation
# file_locs = catalog['file_loc'].str[-29:].head(100).tolist()
file_locs = catalog['file_loc'].str[-29:].head(1000).tolist()

# def crop_image(image, crop_amount):
#     return tf.image.crop_to_bounding_box(image, crop_amount, crop_amount, image.shape[0]-2*crop_amount, image.shape[1]-2*crop_amount).numpy()


# def flip_vertically(image):
#     return tf.image.flip_up_down(image).numpy()


# def denoise_fft(image): #will need to be adjusted, values are arbitrary
#     # Convert to frequency domain
#     im_fft = fftpack.fft2(image)
#     # Remove frequencies below a threshold
#     keep_fraction = 0.1
#     im_fft2 = im_fft.copy()
#     r, c = im_fft2.shape
#     im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
#     im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
#     # Convert back to time domain
#     im_new = fftpack.ifft2(im_fft2).real
#     return im_new

def crop_image(image, crop_size):
    """Crop the image to the center square of size crop_size."""
    width, height = image.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    return image.crop((left, top, right, bottom))

def flip_vertically(image):
    """Flip the image vertically."""
    return image.transpose(Image.FLIP_TOP_BOTTOM)

def flip_horizontally(image):
    if random.choice([True, False]):
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def rotate_image(image):
    angles = [45, 90, -45, -90]
    return image.rotate(random.choice(angles))

def denoise_image(image):
    """Denoise the image using a median filter."""
    image_array = np.asarray(image)
    denoised_image_array = filters.median(image_array)
    return Image.fromarray(denoised_image_array)

def add_noise(image):
    # Convert to numpy array for noise operations
    image_array = np.asarray(image)
    noisy_image = random_noise(image_array)
    # Convert back to Image
    return Image.fromarray((noisy_image * 255).astype(np.uint8))

def apply_random_augmentation(image):
    aug_methods = {
        'flip_h': lambda x: (flip_horizontally(x), 'flip_h'),
        'flip_v': lambda x: (flip_vertically(x), 'flip_v'),
        'rotate': lambda x: (rotate_image(x), 'rotate'),
        'noise': lambda x: (add_noise(x), 'noise'),
        'denoise': lambda x: (denoise_image(x), 'denoise')
    }
    aug_choice = random.choice(list(aug_methods.keys()))
    return aug_methods[aug_choice](image)


for idx, rel_path in enumerate(file_locs):
    full_path = os.path.join(root_dir, rel_path)
    image = Image.open(full_path)
    
    aug_description = 'orig'  # Default description for no augmentation
    if idx >= 20:  # Apply augmentations only to 80% of the images
        image, aug_description = apply_random_augmentation(image)
    
    # Save the augmented image with a descriptive filename
    aug_image_filename = f"aug_image_{idx}_{aug_description}.jpg"
    aug_image_path = os.path.join(aug_images_dir, aug_image_filename)
    image.save(aug_image_path)
    
    
    
    
    
    
    
    