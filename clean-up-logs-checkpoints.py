import os, shutil

# deletes the logs and checkpoints directories

root = "/Users/malachy/Documents/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/"
logpath = os.path.join(root, "logs/")
checkpointpath = os.path.join(root, "checkpoints/")

if os.path.exists(logpath):
    shutil.rmtree(logpath)
    print("\033[32mlogs/ deleted! \033[0m")
else:
    print("\033[31mlogs/ does not exist! \033[0m")
      
if os.path.exists(checkpointpath):
    shutil.rmtree(checkpointpath)
    print("\033[32mcheckpoints/ deleted! \033[0m")
else:
    print("\033[31mcheckpoints/ does not exist! \033[0m")