from dataframe.keys import get_keys, class_labels_to_question
import argparse, os, json, sklearn  # noqa: E401
import sklearn.metrics 
import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd

def build_paths(path, names):
    predictions, keys = names
    predictions_path, keys_path = os.path.join(path, predictions), os.path.join(path, keys)
    return predictions_path, keys_path

def read_json(path):
    """
    Read the JSON file
    
    args:
    path (str): the path to the JSON file
    
    returns:
    data (dict): the JSON file
    """
    with open(path, 'r') as f:
        data = json.load(f)
        
    return data

def json_to_pandas(json, keys, path):
    """
    Convert the JSON file to a pandas dataframe
    
    args:
    json (dict): the JSON file to be converted
    
    returns:
    df (pd.DataFrame): the pandas dataframe
    """
    data_list = []
    for image_path, (true_labels, predicted_labels) in json.items():
        
        row_dict = {"image_path": image_path}
        
        for i, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
                     
            row_dict[f"true_label_{i}"] = true_label
            row_dict[f"predicted_label_{i}"] = predicted_label

        data_list.append(row_dict)
        
    df = pd.DataFrame(data_list)
    
    new_names = keys
    new_truth_names_dict = {f"true_label_{i}": f"TRUTH_{new_names[i]}" for i in range(len(new_names))}
    df = df.rename(columns=new_truth_names_dict)
    new_pred_names_dict = {f"predicted_label_{i}": f"PREDICTED_{new_names[i]}" for i in range(len(new_names))}
    df = df.rename(columns=new_pred_names_dict)
    
    for col in df.columns: # Round the predicted labels
        if col.startswith('PREDICTED_'):
            df[col] = df[col].apply(lambda x: 0.0 if x < 0.5 else 1.0)
        
    print(df)

    df.to_parquet(os.path.join(path, "predictions.parquet"))  # save dataframe as parquet file
    return df

def confusion_matrix(true_labels, predicted_labels):
    """
    true_labels = [[true_labels], [true_labels], ...]
    predicted_labels = [[predicted_labels], [predicted_labels], ...]
    
    args:
    true_labels (list): a list of true labels
    predicted_labels (list): a list of predicted labels
    
    returns:
    cm (np.array): the confusion matrix
    """
    # flatten the arrays to 1D
    true_labels = np.concatenate(true_labels)
    print(true_labels.shape)
    predicted_labels = np.concatenate(predicted_labels)
    print(predicted_labels.shape)
    
    cm = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
    
    print(cm)
    
def main(path, names):
    
    predictions_path, keys_path = build_paths(path, names)
    predictions_json = read_json(predictions_path)
    keys = read_json(keys_path)
    keys = get_keys(keys)
    predictions_df = json_to_pandas(predictions_json, keys, path)

if __name__ == "__main__":
    """run `python cfmatrix.py --help` for argument information"""
    parser = argparse.ArgumentParser(prog = "cfmatrix.py")
    
    parser.add_argument("--path",           type = str,             default = "/Users/malachy/Documents/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/logs/predictions/predictions-ResNet50-2024-05-20-22-19-41",                                         help = "path to the folder containing the predictions")
    parser.add_argument("--names",          type = tuple,           default = ('predictions.json', 'keys.json'),              help = "names of the json files to read")
    
    args = parser.parse_args()
          
    if args.path and os.path.exists(args.path): 
        kwargs = vars(args)
        main(**kwargs)            
    else:
        raise ValueError("\033[31mYou must provide predictions to use! \n\nHint: `python cfmatrix.py --path '<root>/logs/<predictions>/`\nrun `python  cfmatrix.py.py --help` for argument information.\033[0m")