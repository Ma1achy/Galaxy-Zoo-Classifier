import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras as keras
import time as time
import datetime as datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#================================================================ CLASSES AND FUNCTIONS ===================================================================
class ConvolutionalNeuralNetwork():
    def __init__(self, display_summary, architecture, learning_rate, weight_reg, representation, input_shape, input_dim):
        """
        Construct a Convolutional Neural Network of a specified architecture.
        
        args:
        display_summary (bool): display the summary of the network
        architecture (str): the architecture of the network
        learning_rate (float): the learning rate of the network
        weight_reg (str): the weight regularization of the network
        train_class_labels (list): the class labels of the training data
        test_class_labels (list): the class labels of the test data
        representation (str): the representation of the data, i.e one-hot or vote fraction
        input_dim (int): the input dimension of the data
        """
        
        self.display_summary = display_summary
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.weight_reg = weight_reg
        
        self.representation = representation
        self.label_length = len(get_keys(representation))
        
        self.input_x, self.input_y = input_shape
        self.input_dim = input_dim
        
        self.network = self.__init_network(display_summary, architecture)
        
        self.train_history = None
        self.test_history = None
    #------------------------------------------------------------------------- NETWORK INITIALIZATION -----------------------------------------------------------------------    
    def __init_network(self, display_summary: bool = False, architecture: str = "216x64") -> tf.keras.Model:
        """
        Initalise a Convolutional Neural Network of a specified architecture using the Keras Sequential API.
        
        args:
        display_summary (bool): display the summary of the network
        architecture (str): the architecture of the network
        """
        lr = self.learning_rate
        
        # Weight regularization
        if self.weight_reg is None:
            wr = None
        else:
            wr = tf.keras.regularizers.l2(self.weight_reg)
        
        if architecture == '128x64xn':
            
            network = Sequential()
            network.add(Conv2D(filters = 32, kernel_size = (7, 7), strides = 3, activation = 'relu', kernel_regularizer = wr, input_shape = (self.input_x, self.input_y, self.input_dim), name = "Convolutional_1")) # input_shape doesn't work, issue with combining tuple self.input_shape and int self.input_dim
            network.add(MaxPooling2D(pool_size = (2,2)))
            network.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, activation = 'relu', kernel_regularizer = wr, name = "Convolutional_2"))
            network.add(MaxPooling2D(pool_size = (2,2)))
            network.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, activation = 'relu', kernel_regularizer = wr, name = "Convolutional_3"))
            network.add(Flatten())
            network.add(Dense(units = 128, activation = 'relu', kernel_regularizer = wr, name = "Dense_1"))
            network.add(Dropout(0.5))
            network.add(Dense(units = 64, activation = 'relu', kernel_regularizer = wr, name = "Dense_2"))
            network.add(Dropout(0.5))
            network.add(Dense(self.label_length, activation = None, kernel_regularizer = wr)) 
            
            network.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate = lr, epsilon = 1e-07), metrics = ['accuracy'])
            
        elif architecture == '400xn':
            
            pass
             
        else:
            raise ValueError(f"Invalid architecture parameter provided: {architecture}")
        
        if (display_summary):
            network.summary()
            
        return network
    #------------------------------------------------------------------------- NETWORK TRAINING AND TESTING ---------------------------------------------------------------------------
    def train(self, train_data, validation_data, dataframe, train_batch_size, val_batch_size, val_steps, epochs, initial_epoch):
        """
        Train the network on the training data using model.fit().
        
        args:
        train_data (tf.data.Dataset): the training dataset
        validation_data (tf.data.Dataset): the validation dataset
        dataframe (DataFrame): An instance of DataFrame containing the representation of the training and test data
        train_batch_size (int): the batch size of the training data
        val_batch_size (int): the batch size of the validation data
        val_steps (int): the number of steps to run the validation data
        epochs (int): the number of epochs to train the network
        initial_epoch (int): the initial epoch to start training the network from
        """
        train_data = train_data.map(
            lambda image, label: dataframe.fetch_image_label_pair(image, label),
            num_parallel_calls = tf.data.experimental.AUTOTUNE # allow for tensorflow to dynamically decide the number of parallel calls to make
        ).batch(
            train_batch_size # batch the dataset into batches of size batch_size
        ).prefetch(
            tf.data.experimental.AUTOTUNE # allow for prefetching to parallelize the loading of the images and labels with the training of the network
        )
        
        validation_data = validation_data.map(
            lambda image, label: dataframe.fetch_image_label_pair(image, label),
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        ).batch(
            val_batch_size
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        
        self.train_history = self.network.fit(
            train_data,
            epochs = epochs,
            verbose = 1,
            callbacks = None,
            validation_data = validation_data,
            shuffle = True,
            initial_epoch = initial_epoch,
            steps_per_epoch = None,
            validation_steps = val_steps,
            validation_batch_size = val_batch_size,
            validation_freq = 1,
            max_queue_size = 10,
            workers = 1,
            use_multiprocessing = True
        )
        
    def test(self, test_data, dataframe, test_batch_size, test_steps):
        """
        Evaluate the network on the test data using model.evaluate().
        
        args:
        test_data (tf.data.Dataset): the test dataset
        dataframe (DataFrame): An instance of DataFrame containing the representation of the training and test data
        test_batch_size (int): the batch size of the test data
        test_steps (int): the number of steps to run the test data
        """
        
        test_data = test_data.map(
            lambda image, label: dataframe.fetch_image_label_pair(image, label),
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        ).batch(
            test_batch_size
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        
        self.test_history = self.network.evaluate(
            test_data,
            batch_size = test_batch_size,
            verbose = 1,
            steps = test_steps,
            callbacks = None,
            max_queue_size = 10,
            workers = 1,
            use_multiprocessing = True,
            return_dict = True
        )
        
    #----------------------------------------------------------------- NETWORK WEIGHTS AND BIASES SAVING AND LOADING ----------------------------------------------------------------   
    def save(self, checkpoint_path):
        """
        Save the network weights and biases to a file.
        
        args:
        checkpoint_path (str): the path to the directory to save the network weights and biases
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(checkpoint_path, 'checkpoint-' + timestamp)
        self.network.save_weights(path)
        
    
    def load(self, checkpoint_path, timestamp):
        """
        Load the network weights and biases from a file.
        
        args:
        checkpoint_path (str): the path to the directory to load the network weights and biases
        """
        path = os.path.join(checkpoint_path, 'checkpoint-' + timestamp)
        self.network.load_weights(path)
    
class DataFrame():
    def __init__(self, dataset_root, train_catalog, test_catalog, representation, input_shape, input_dim, seed):
        """
        Construct a DataFrame containing the represenation of the training and test data in the required format, i.e one-hot or vote fraction.
        
        args:
        dataset_root (str): path to the dataset root directory
        train_catalog (str): name of the training catalog file
        test_catalog (str): name of the test catalog file
        keys (list): list of the keys to be used when acessing the class labels
        representation (str): the representation of the data, i.e one-hot or vote fraction
        input_shape (tuple): the input shape of the data, x, y
        input_dim (int): the input dimension of the data, i.e 3 for RGB image
        """
        
        self.dataset_root = dataset_root
        
        self.train_parquet_file_path = os.path.join(self.dataset_root, train_catalog)
        self.test_parquet_file_path = os.path.join(self.dataset_root, test_catalog)
        
        self.representation = representation
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.keys = get_keys(representation)
        
        self.train_dataset = self.__init_dataset(self.keys, self.train_parquet_file_path)
        self.test_dataset = self.__init_dataset(self.keys, self.test_parquet_file_path)
        
        tf.random.set_seed(seed)
        
    #--------------------------------------------------------------------- DATASET(S) INITIALIZATION ---------------------------------------------------------------------------------------
    def __init_dataset(self, keys, catalog_path):
        """
        Initialise a dataset, containing paths to the image files and the labels in the required representation.
        
        args:
        keys (list): the keys of the class labels
        catalog_path (str): the path to the catalog file
        
        returns:
        dataset (tf.data.Dataset): a zipped dataset in the required format, i.e one-hot or vote fraction. containing a tuple of a tf.data.Dataset of image paths and a tf.data.Dataset of labels 
        """
        if self.representation == 'one-hot':
            
            pandas_df = pd.read_parquet(catalog_path)
            pandas_df = pandas_df.fillna(0) 
            class_labels = pandas_df['label'] + 1
            one_hot = pd.get_dummies(class_labels).astype(int) # convert the class labels to one-hot encoding
            
            pandas_df['file_loc'] = np.empty(len(pandas_df['id_str']), dtype = str) # delete the information contained in 'file_loc' 
            
            for idx, _ in enumerate(pandas_df['id_str']):  # construct the actual path to the image file and add it to 'file_loc'
                pandas_df.loc[idx, 'file_loc'] = os.path.join(self.dataset_root, 'images/', str(pandas_df['subfolder'][idx]) + '/', str(pandas_df['filename'][idx]))
                
                if not os.path.exists(pandas_df['file_loc'][idx]): # check if the file exists at the constructed path, if not then remove the row from the DataFrame
                    pandas_df = pandas_df.drop(idx)
                    one_hot = one_hot.drop(idx)
                
            image_path_ds = tf.data.Dataset.from_tensor_slices(np.array(pandas_df['file_loc']).reshape((len(pandas_df), 1)))
            labels_ds = tf.data.Dataset.from_tensor_slices(np.array(one_hot))
                
            return tf.data.Dataset.zip((image_path_ds, labels_ds))
            
        elif self.representation == 'vote-fraction':
            
            pandas_df = pd.read_parquet(catalog_path)
            pandas_df = pandas_df.fillna(0)
            
            pandas_df['file_loc'] = np.empty(len(pandas_df['id_str']), dtype = str) 
            
            for idx, _ in enumerate(pandas_df['id_str']):  
                pandas_df.loc[idx, 'file_loc'] = os.path.join(self.dataset_root, 'images/', str(pandas_df['subfolder'][idx]) + '/', str(pandas_df['filename'][idx]))

                if not os.path.exists(pandas_df['file_loc'][idx]): 
                    pandas_df = pandas_df.drop(idx)
            
            image_path_ds = tf.data.Dataset.from_tensor_slices(np.array(pandas_df['file_loc']).reshape((len(pandas_df), 1)))
            labels_ds = tf.data.Dataset.from_tensor_slices(np.array([pandas_df[key] for key in keys]).reshape((len(pandas_df), len(keys))))   
                    
            return tf.data.Dataset.zip((image_path_ds, labels_ds))
        else:
            raise ValueError(f"Training DataFrame: Invalid representation parameter provided: {self.representation}")
    
    #--------------------------------------------------------------------------------- IMAGE PROCESSING -------------------------------------------------------------------------------------
    def fetch_image_label_pair(self, image_path, label):
        """
        Fetches an image and its corresponding label from the dataset.
        
        args:
        image_path (tf.Tensor): the path to the image file
        label (tf.Tensor): the label of the image
        
        returns:
        (tuple): of the image and its corresponding label
            image (tf.Tensor): the requested image
            label (tf.Tensor): the requested label
        """
        image_path = tf.squeeze(image_path)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels = self.input_dim)
        image = tf.image.convert_image_dtype(image, tf.float32) /255  # Convert image dtype from tf.uint8 to tf.float32 to allow for image normalisation
        image = tf.image.resize(image, self.input_shape) 
        
        return image, label
    
    #TODO: implement image augmentation methods here, i.e rotation, flipping, scaling, translation, brightness, contrast, saturation, hue, denoising, adding noise, reducing resolution, blurring
    #----------------------------------------------------------------------------- VALIDATION DATA SPLITTING ---------------------------------------------------------------------------------
    def get_train_val_split(self, data, val_fraction, shuffle, seed):
        """
        Split the training dataset into a training and validation dataset.
        
        args:
        data (tf.data.Dataset): the input dataset to be split
        val_fraction (float): the fraction of the dataset to be used for validation, range: [0, 1]
        shuffle (bool): shuffle the dataset before splitting
        seed (int): the seed for the random number generator
        
        returns: 
        (tuple): of training and validation datasets
            train_split (tf.data.Dataset): the training dataset
            validation_split (tf.data.Dataset): the validation dataset
        """
        train_split, validation_split = tf.keras.utils.split_dataset(data, left_size = 1 - val_fraction, right_size = val_fraction, shuffle = shuffle, seed = seed)

        return train_split, validation_split
    
    def get_kfolds(self):
        pass
    
def get_keys(representation):
    """
    Obtain the keys of the class labels when accessing the class labels from a Pandas DataFrame.
    
    args:
    representation (str): the representation of the data, i.e one-hot or vote fraction
    
    returns:
    keys (list): the keys of the class labels
    """
    if representation == 'one-hot':
        keys = [
        0, 
        1, 
        2, 
        4, 
        5, 
        6, 
        7
    ] 
        return keys     
    elif representation == 'vote-fraction':
        keys = [
            'smooth-or-featured-gz2_smooth_fraction',
            'smooth-or-featured-gz2_featured-or-disk_fraction',
            'smooth-or-featured-gz2_artifact_fraction',
            'disk-edge-on-gz2_yes_fraction',
            'disk-edge-on-gz2_no_fraction',
            'bar-gz2_yes_fraction',
            'bar-gz2_no_fraction',
            'has-spiral-arms-gz2_yes_fraction',
            'has-spiral-arms-gz2_no_fraction',
            'bulge-size-gz2_no_fraction',
            'bulge-size-gz2_just-noticeable_fraction',
            'bulge-size-gz2_obvious_fraction',
            'bulge-size-gz2_dominant_fraction',
            'something-odd-gz2_yes_fraction',
            'something-odd-gz2_no_fraction',
            'how-rounded-gz2_round_fraction',
            'how-rounded-gz2_in-between_fraction',
            'how-rounded-gz2_cigar_fraction',
            'bulge-shape-gz2_round_fraction',
            'bulge-shape-gz2_boxy_fraction',
            'bulge-shape-gz2_no-bulge_fraction',
            'spiral-winding-gz2_tight_fraction',
            'spiral-winding-gz2_medium_fraction',
            'spiral-winding-gz2_loose_fraction',
            'spiral-arm-count-gz2_1_fraction',
            'spiral-arm-count-gz2_2_fraction',
            'spiral-arm-count-gz2_3_fraction',
            'spiral-arm-count-gz2_4_fraction',
            'spiral-arm-count-gz2_more-than-4_fraction',
            'spiral-arm-count-gz2_cant-tell_fraction',
        ]
        return keys
    else:
        raise ValueError(f"Invalid representation parameter provided: {representation}")
        
#============================================================================== MAIN FUNCTION ============================================================================
def main(dataset_root, train_catalog, test_catalog, representation, input_shape, input_dim, display_summary, architecture, learning_rate, weight_reg, batch_size, epochs, seed):
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        for gpu in gpus:
            os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*7.5)])
        
    df = DataFrame(dataset_root, train_catalog, test_catalog, representation, input_shape, input_dim, seed)
    CNN = ConvolutionalNeuralNetwork(display_summary, architecture, learning_rate, weight_reg, representation, input_shape, input_dim)
    
    train_split, val_split = df.get_train_val_split(df.train_dataset, val_fraction = 0.2, shuffle = True, seed = seed)
    CNN.train(train_data= train_split, validation_data = val_split, dataframe = df, train_batch_size = batch_size, val_batch_size = 16, val_steps = 2, epochs = epochs, initial_epoch = 0)
    
    #TODO: implement autosaving and loading of the network weights and biases
    # implement kfold cross validation with tf.datasets or sklearn
    # implement training and testing metrics, i.e accuracy, precision, recall, f1-score, confusion matrix
    # implement plotting of the training and testing metrics

#===================================================================== COMMAND LINE ARGUMENT(S) PARSER ===================================================================  
if __name__ == "__main__":
    """run `python network.py --help` for argument information""" 
    # ensure you are in the correct directory where py file is located, can change directory by running `cd "<directory path>"`
    # i.e run `cd "C:/Users/Ma1a_/Desktop/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/main"` to change directory
    
    parser = argparse.ArgumentParser(prog = "network.py")
    
    parser.add_argument("-dim",       "--inputdim", type = int,   default = 3,          help = "(int) The input dimension of the data")
    parser.add_argument("-ixy",     "--inputshape", type = tuple, default = (424, 424), help = "(tuple) The input shape of the data, i.e (424, 424) for 424x424 images")
    
    parser.add_argument("-r",             "--root", type = str,   default = r"C:/Users/Ma1a_/Desktop/3rd Year Project/Project-72-Classifying-cosmological-data-with-machine-learning/galaxyzoo2-dataset", help = "(str) The path to the dataset root directory")
    parser.add_argument("-l",           "--labels", type = str,   default = r"gz2_train_catalog.parquet", help = "(str) The name of the training catalog file")
    parser.add_argument("-tl",      "--testlabels", type = str,   default = r"gz2_test_catalog.parquet",  help = "(str) The name of the test catalog file") 
    parser.add_argument("-rep", "--representation", type = str,   default = "vote-fraction",              help = "(str) The representation of the data, i.e one-hot or vote-fraction")
    
    parser.add_argument("-sum",        "--summary", type = bool,  default = False,      help = "(bool) Display the summary of the network")
    parser.add_argument("-a",     "--architecture", type = str,   default = "128x64xn", help = "(str) The architecture of the network")
    parser.add_argument("-lr",    "--learningrate", type = float, default = 0.001,      help = "(float) The learning rate of the network")
    parser.add_argument("-wr",       "--weightreg", type = str,   default = None,       help = "(str) The weight regularization of the network")
    parser.add_argument("-bs",       "--batchsize", type = int,   default = 32,         help = "(int) The batch size of the network")
    parser.add_argument("-e",           "--epochs", type = int,   default = 500,        help = "(int) The number of epochs to train the network")
    parser.add_argument("-s",             "--seed", type = int,   default = 0,          help = "(int) The seed for the random number generator")
    
    args = parser.parse_args()

    main(
        input_dim = args.inputdim,
        input_shape = args.inputshape,
        dataset_root = args.root,
        train_catalog = args.labels,
        test_catalog = args.testlabels,
        representation = args.representation,
        display_summary = args.summary,
        architecture = args.architecture,
        learning_rate = args.learningrate,
        weight_reg = args.weightreg,
        batch_size = args.batchsize,
        epochs = args.epochs,
        seed = args.seed
    )       
