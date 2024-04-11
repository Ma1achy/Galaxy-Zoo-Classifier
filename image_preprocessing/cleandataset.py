# go through the dataset and remove any images from the dataframe and folders that do not have a file path that exists
# remove any images from the dataframe and folders where the bottom half of the image is only black
import os, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_missing_paths(dataset_root, catalog_path):
    """
    Remove missing paths from the catalog dataframe and return a new dataframe with the missing paths removed
    
    Args:
    dataset_root (str): path to the root of the dataset
    catalog_path (str): path to the catalog dataframe
    
    Returns:
    pandas_df (pd.DataFrame): new dataframe with missing paths removed
    """
    pandas_df = pd.read_parquet(catalog_path)
    
    missing_indices = []
    missing_paths = 0

    for idx, _ in enumerate(pandas_df['id_str']):
        file_loc = os.path.join(dataset_root, 'images/', str(pandas_df['subfolder'][idx]) + '/', str(pandas_df['filename'][idx])) 
        pandas_df.at[idx, 'file_loc'] = file_loc  # write new file_loc to file_loc column

        if not os.path.exists(file_loc): 
            missing_indices.append(idx)
            missing_paths += 1
            print(f"Missing path: {pandas_df['subfolder'][idx]}/{pandas_df['filename'][idx]}", end = '\r', flush = True)

    print("\n")
    for idx in missing_indices:
        pandas_df = pandas_df.drop(idx)

    pandas_df = pandas_df.reset_index(drop=True)

    print("Missing paths removed")
    print(f"Number of missing paths removed: {missing_paths}")

    return pandas_df
    
def remove_error_msg_images(pandas_df, show_images, delete_paths):
    """
    Remove error message images from the dataset
    
    Args:
    pandas_df (pd.DataFrame): dataframe to remove error message images from
    show_images (bool): show images that are removed
    delete_paths (bool): delete the paths of the removed images
    
    Returns:
    pandas_df (pd.DataFrame): new dataframe with error message images removed
    """
    
    print("Removing error message images")
    erorr_count = 0
    removed_paths = []
    
    for idx, _ in enumerate(pandas_df['id_str']):
        print(f"Reading: {pandas_df.loc[idx, 'subfolder']}/{pandas_df.loc[idx, 'filename']}", end = '\r', flush = True)
        image_path = os.path.join(dataset_root, 'images/', str(pandas_df.loc[idx, 'subfolder']) + '/', str(pandas_df.loc[idx, 'filename']))
        
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            cropped_image = image[image.shape[0]//2:, :]
        
            if np.all(cropped_image == 0):
                removed_paths.append(image_path)
                print(f"Error message image found: {pandas_df.loc[idx, 'subfolder']}/{pandas_df.loc[idx, 'filename']}")
                if show_images:
                    plt.imshow(image)
                    plt.title(f"{pandas_df['subfolder'][idx]}/{pandas_df['filename'][idx]}")
                    plt.show()
                
                erorr_count += 1 
    
    for image_path in removed_paths:
        if delete_paths:
            print("Removing: ", image_path, end = '\r', flush = True) 
            try:
                os.remove(image_path)
                print(f"Deleted: {image_path}")
            except Exception as e:
                print(f"Failed to delete {image_path}: {e}")
        
    pandas_df = pandas_df.reset_index(drop=True)         
        
    print(f"Number of error message images removed: {erorr_count}")
            
    return pandas_df

def build_new_catalog(pandas_df, catalog_path):
    """
    Build a new catalog dataframe and save it to a parquet file
    
    Args:
    pandas_df (pd.DataFrame): dataframe to build new catalog from
    catalog_path (str): path to save the new catalog
    """
    print("Building new catalog")
    os.remove(catalog_path)
    pandas_df = pandas_df.reset_index(drop=True)
    pandas_df['iauname'] = pandas_df['iauname'].astype(str)
    pandas_df['summary'] = pandas_df['summary'].astype(str)
    pandas_df.to_parquet(catalog_path)
    print(f"New catalog saved to: {catalog_path}")

dataset_root = "/Users/malachy/Documents/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset-processed"
train_catalog = "/Users/malachy/Documents/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset-processed/gz2_train_catalog.parquet"
test_catalog = "/Users/malachy/Documents/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset-processed/gz2_test_catalog.parquet"

paths_dataframe = remove_missing_paths(dataset_root, train_catalog)
processed_dataframe = remove_error_msg_images(paths_dataframe, show_images = False, delete_paths = True)
build_new_catalog(processed_dataframe, catalog_path = train_catalog)

paths_dataframe = remove_missing_paths(dataset_root, test_catalog)
processed_dataframe = remove_error_msg_images(paths_dataframe, show_images = False, delete_paths = True)
build_new_catalog(processed_dataframe, catalog_path = test_catalog)