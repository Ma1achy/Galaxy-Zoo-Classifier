import os, cv2
import pandas as pd
import numpy as np
import time

# duplicate images in dataset by applying augmentations to them.
# create new subfolder with A appended to the start of the subfolder name and save the augmented images there.
# do this for all images in the dataset, 4 times for each image, add an extra A to the subfolder name each time.
# append the new images to the end of the catalog, keeping the original data associated with the image, but replace the file_loc with the new file_loc.
# can apply multiple augmentations to the same image, i.e rotation by random angle, flipping, zoom, adding noise, de-noising ect.

def read_parquet(path):
    """
    Read the parquet file at the path.
    
    args:
    path (str): The path to the parquet file.
    
    returns:
    pandas_df (pd.DataFrame): The dataframe containing the data from the parquet file.
    """
    return pd.read_parquet(path)

def open_image(image_path):
    """
    Open the image at the image_path.
    
    args:
    image_path (str): The path to the image.
    
    returns:
    image (tf.Tensor): The image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_augmentations(image, RNG):
    """
    Get a random set of 3 augmentations to apply to the image.
    
    args:
    image (tf.Tensor): The image to apply augmentations to.
    RNG (np.random.Generator): The random number generator.
    
    returns:
    image (tf.Tensor): The image with the augmentations applied.
    aug_1 (str): The first augmentation applied.
    aug_2 (str): The second augmentation applied.
    aug_3 (str): The third augmentation applied.
    """
    
    augmentations = np.random.choice(['None', 'Flip', 'Rotation', 'Scale', 'Blurr', 'Noise', 'Translation'], 3, replace = False)
    aug_1, aug_2, aug_3 = augmentations
    
    for aug in augmentations:
        match aug:
            case 'None':
                image = no_augmentations(image)
            case 'Flip':
                image = random_flip(image, RNG)
            case 'Rotation':
                image = random_rotation(image, RNG)
            case 'Scale':
                image = random_zoom(image, RNG)
            case 'Hue':
                image = random_hue(image, RNG)
            case 'Blurr':
                image = random_blurr(image, RNG)
            case 'Noise':
                image = noise_denoise(image, RNG)
            case 'Translation':
                image = random_translation(image, RNG)
    
    image = cv2.convertScaleAbs(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, aug_1, aug_2, aug_3

def no_augmentations(image):
    """
    Do not apply any augmentations to the image.
    
    args:
    image (tf.Tensor): The image to not apply augmentations to.
    
    returns:
    image (tf.Tensor): The original image.
    """
    return image

def random_flip(image, RNG):
    """
    Flip the image along a random axis.
    axis = 0 -> flip up-down
    axis = 1 -> flip left-right
    
    args:
    image (np.array): The image to flip.
    RNG (np.random.Generator): The random number generator.
    
    returns:
    image (np.array): The flipped image.
    """
    axis = RNG.integers(0, 1, endpoint = True)
    image = cv2.flip(image, axis)
    return image
    
def random_rotation(image, RNG):
    """
    Rotate the image by a random angle between -180 and 180 degrees.
    
    args:
    image (tf.Tensor): The image to rotate.
    RNG (np.random.Generator): The random number generator.
    
    returns:
    image (np.array): The rotated image
    """
    rows, cols, dim = image.shape
    angle = RNG.uniform(-180, 180)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image
    
def random_zoom(image, RNG):
    """
    Zoom the image by a random value between -0.5 and 0.5.
    Ensure the aspect ratio of the zoomed image is square.
    
    args:
    image (np.array): The image to zoom.
    RNG (np.random.Generator): The random number generator.
    
    returns:
    image (np.array): The zoomed image.
    """
    scale = RNG.uniform(-0.75, 1.5)
    rows, cols, dim = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, scale)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def random_hue(image, RNG):
    """
    Change the hue of the image by a random value between -180 and 180.
    
    args:
    image (np.array): The image to change the hue of.
    RNG (np.random.Generator): The random number generator.
    
    returns:
    image (np.array): The image with the hue changed.
    """
    delta = RNG.uniform(-180, 180)  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image)
    h = np.clip(h + delta, 0, 255)
    image = cv2.merge((h, s, v))
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def random_blurr(image, RNG):
    """
    Apply a random blurring filter to the image.
    
    args:
    image (np.array): The image to apply the blurring filter to.
    RNG (np.random.Generator): The random number generator.
    
    returns:
    image (np.array): The image with the blurring filter applied.
    """
    k = RNG.choice([3, 5, 7])
    image = cv2.GaussianBlur(image, ksize = (k, k), sigmaX = 0, sigmaY = 0)
    return image

def noise_denoise(image, RNG):
    """
    Add random noise to the image or denoise the image.
    50% chance of adding noise, 50% chance of denoising the image.
    
    args:
    image (np.array): The image to add noise to or denoise.
    RNG (np.random.Generator): The random number generator.
    
    returns:
    image (np.array): The image with noise added or denoised.
    """
    choice = RNG.integers(0, 1, endpoint=True)
 
    if choice == 0:
        noise = np.random.normal(0, 1, image.shape)
        image = image + noise
        image = np.clip(image, 0, 255)
        return image
    else:
        image = cv2.fastNlMeansDenoisingColored(image, h = 10, hColor = 10, templateWindowSize = 7, searchWindowSize = 21)
        return image

def random_translation(image, RNG):
    """
    Translate the image by a random value between -0.5 and 0.5.
    
    args:
    image (np.array): The image to translate.
    RNG (np.random.Generator): The random number generator.
    
    returns:
    image (np.array): The translated image.
    """
    rows, cols, dim = image.shape
    tx = RNG.uniform(-0.25, 0.25) * cols
    ty = RNG.uniform(-0.25, 0.25) * rows
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image

def augment_images(pandas_df, max_iter, dataset_root, catalog, RNG):
    """
    Augment the images in the pandas_df by applying random augmentations to them.
    Save the augmented images to a new subfolder in the images directory.
    Append the new image data to the end of the pandas_df.
    Repeat this process max_iter times.
    
    Parameters:
    pandas_df (pd.DataFrame): The dataframe containing the image data.
    max_iter (int): The number of times to augment the images.
    dataset_root (str): The root directory of the dataset.
    catalog (str): The name of the catalog.
    RNG (np.random.Generator): The random number generator.
    """
    
    times = []
    
    for iter in range(1, max_iter + 1):

        new_pandas_df = pd.DataFrame()
        
        for idx, _ in enumerate(pandas_df['file_loc']):
            
            start_time = time.time()
            image = pandas_df.loc[idx, 'file_loc']
            image_data = pandas_df.loc[idx].copy()
            image = open_image(image)
            image, aug_1, aug_2, aug_3  = get_augmentations(image, RNG)
            
            image_subfolder = f"Augmented-{iter}-{image_data['subfolder']}"
            image_name = f"A{iter}-{aug_1}-{aug_2}-{aug_3}-{image_data['filename']}"
            
            sub_folder_path = os.path.join(dataset_root, 'images/', image_subfolder)
            
            if not os.path.exists(sub_folder_path):
                os.mkdir(sub_folder_path)
            
            image_path = os.path.join(sub_folder_path, image_name)
        
            image_data['subfolder'] = image_subfolder
            image_data['filename'] = image_name
            image_data['file_loc'] = image_path
            
            new_pandas_df = new_pandas_df._append(image_data, ignore_index = True)
            
            cv2.imwrite(image_data['file_loc'], image)
            
            end_time = time.time()
            iter_time = end_time - start_time
            times.append(iter_time)
            
            if len(times) > 1024:
                times = times[:len(times)//2]

            if times == []:
                mean_time = None
            else:
                mean_time = np.mean(times)
            
            print(f"iter: {iter}/{max_iter} \t | {idx}/{len(pandas_df['file_loc'])} \t | ETA: {mean_time * (len(pandas_df['file_loc']) - idx):.2e}s \t | [{image_subfolder}/{image_name}] \t\t\t", end = '\r', flush
            = True)
            
        # save the new_pandas_df to file
        new_pandas_df.to_parquet(f"{dataset_root}/augmented_{iter}_{catalog}")

dataset_root = "/Users/malachy/Documents/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset-processed"
train_catalog = "gz2_train_catalog.parquet"
test_catalog = "gz2_test_catalog.parquet"

train_catalog_path = os.path.join(dataset_root, train_catalog)
test_catalog_path = os.path.join(dataset_root, test_catalog)

train_df = read_parquet(train_catalog_path)
test_df = read_parquet(test_catalog_path)
max_iter = 1

seed = 1
RNG = np.random.default_rng(seed)

augment_images(pandas_df = train_df, max_iter = max_iter, dataset_root = dataset_root, catalog = train_catalog, RNG = RNG)
augment_images(pandas_df = test_df, max_iter = max_iter, dataset_root = dataset_root, catalog = test_catalog, RNG = RNG)

# should probably manually delete the duplicate catalogs created by the augmentation process, but
# I won't make a script to do that as its also handy to keep them around incase something breaks.