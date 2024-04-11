from galaxy_datasets import gz2 
import os 

#"pip install galaxy-datasets" to install galaxy-datasets package.
# if this doesn't work try "-m pip install galaxy-datasets" 

#This should just work if you run this.

datasetpath = 'galaxyzoo2-dataset' #change this to your desired path for the dataset

root = r'D:/3rd year project/Project-72-Classifying-cosmological-data-with-machine-learning' #change this to your root directory (should be this by default)
datasetpath = os.path.join(root, 'galaxyzoo2-dataset') #change this to your desired path for the dataset

if not os.path.exists(datasetpath):
    os.makedirs(datasetpath)

catalog, labels = gz2(
    root= datasetpath,
    train= True,
    download= True
)