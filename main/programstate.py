from packages.imports import os, json, np, tf

global_state_dict = {} # contains the variables that define the state of the program, used for saving and loading

def write_state(path):
    """
    Write the global state dictionary to a json file
    
    Args:
    path (str): path to save the json file
    """
    if not os.path.exists(os.path.join(path, "metadata.json")):
        os.makedirs(os.path.dirname(os.path.join(path, "metadata.json")), exist_ok = True)
            
    with open(os.path.join(path, "meta.json"), 'w') as f:
        json.dump(global_state_dict, f, indent = 4)
                
def read_json(path):
    """
    Read a json file and return the dictionary
    
    Args:
    path (str): path to the json file
    
    Returns:
    dict (dict): dictionary from the json file
    """
    
    print(f"\033[33mReading json from: [{path}]...\033[0m")
        
    with open(path, "r") as file:
        dict = json.load(file)
    
    return dict