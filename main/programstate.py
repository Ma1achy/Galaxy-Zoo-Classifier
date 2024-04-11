from packages.imports import os, json, np, tf

global_state_dict = {}

def write_state(path):
    
    if not os.path.exists(os.path.join(path, "metadata.json")):
        os.makedirs(os.path.dirname(os.path.join(path, "metadata.json")), exist_ok = True)
            
    with open(os.path.join(path, "meta.json"), 'w') as f:
        json.dump(global_state_dict, f, indent = 4)
                
def read_json(path):
    
    print(f"\033[33mReading json from: [{path}]...\033[0m")
        
    with open(path, "r") as file:
        dict = json.load(file)
    
    return dict