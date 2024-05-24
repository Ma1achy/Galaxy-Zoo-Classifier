import argparse, os, json  # noqa: E401
import pandas as pd

def combine_catalogs(root, new_catalog, *catalogs):
    """
    Combine multiple catalogs into a single catalog file and save it as a parquet file.
    
    Args:
    root (str): The root directory of the catalog files.
    new_catalog (str): The name of the new catalog file.
    *catalogs (str): The names of the catalog files to combine.
    """
    
    combined_data = pd.DataFrame()
    parquets = []
    
    print("\033[94mCombining catalogs...\033[0m")
    for catalog in catalogs:
        file_path = os.path.join(root, catalog)
        parquets.append(pd.read_parquet(file_path))

    combined_data = pd.concat(parquets, ignore_index = True)
    combined_data.to_parquet(os.path.join(root, new_catalog))
    print(f"\033[32mCombined catalogs saved as [{new_catalog}]\033[0m")

def read_json(root, json_file):
    """
    Read a JSON file and return the contents as a dictionary.
    
    Args:
    json_file (str): The path to the JSON file.
    
    Returns:
    dict: The contents of the JSON file.
    """
    json_file = os.path.join(root, json_file)
    
    with open(json_file, "r") as file:
        return json.load(file)
                         
def main(root, new_train_catalog, new_test_catalog, train_json, test_json):
    
    train_dict = read_json(root, train_json)
    test_dict = read_json(root, test_json)
    
    combine_catalogs(root, new_train_catalog, *train_dict)
    combine_catalogs(root, new_test_catalog, *test_dict)

if __name__ == "__main__":
    """Run `python combine-catalogs.py --help` for argument information"""
    parser = argparse.ArgumentParser(prog = "combine-catalogs.py")
    
    parser.add_argument("-r",        "--root",                  type = str,    default = "/home/malachy/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset-augmented",   help = "(str) The path to the root directory")
    parser.add_argument("-ntrain",   "--new_train_catalog",     type = str,    default = "combined-train_catalog.parquet",      help = "(str) The name of the new catalog file")
    parser.add_argument("-ntest",    "--new_test_catalog",      type = str,    default = "combined-test_catalog.parquet",       help = "(str) The name of the new catalog file")
    parser.add_argument("-ttj",      "--train_json",            type = str,    default = "train-to-combine.json",               help = "(str) The JSON files containing the names of the catalog files to combine")
    parser.add_argument("-tej",      "--test_json",             type = str,    default = "test-to-combine.json",                help = "(str) The JSON files containing the names of the catalog files to combine")
    
    args = parser.parse_args()
    
    kwargs = vars(args)
    main(**kwargs)
    