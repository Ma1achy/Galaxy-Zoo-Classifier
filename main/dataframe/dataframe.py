from packages.imports import os, tf, pd, np, datetime
from dataframe.keys import get_keys, class_labels_to_question

def get_paths(root, dataset_root, checkpointroot, logroot, time_stamp):
    """
    Get the paths to the dataset, checkpoint and log directories.
    
    args:
    root (str): the path to the root directory
    dataset_root (str): the path to the dataset root directory
    checkpointroot (str): the path to the checkpoint root directory
    logroot (str): the path to the log root directory
    time_stamp (str): the date and time to use when naming the paths
    
    returns:
    dataset_root (str): the path to the dataset root directory
    checkpoint_path (str): the path to the checkpoint directory
    log_path (str): the path to the log directory
    """
    if not os.path.exists(os.path.join(root, checkpointroot + f"/{time_stamp}")):
        os.makedirs(os.path.join(root, checkpointroot + f"/{time_stamp}"))
    
    if not os.path.exists(os.path.join(root, logroot + f"/{time_stamp}")):
        os.makedirs(os.path.join(root, logroot + f"/{time_stamp}"))
        
    dataset_root = os.path.join(root, dataset_root)
    checkpoint_path = os.path.join(root, checkpointroot + f"/{time_stamp}")
    log_path = os.path.join(root, logroot + f"/{time_stamp}")
    
    return dataset_root, checkpoint_path, log_path

class DataFrame():
    def __init__(self, dataset_root, train_catalog, test_catalog, classes, binarised_labels, input_shape, input_dim, seed, ds_fraction):
        """
        Construct a DataFrame containing the represenation of the training and test data
        
        args:
        dataset_root (str): path to the dataset root directory
        train_catalog (str): name of the training catalog file
        test_catalog (str): name of the test catalog file
        keys (list): list of the keys to be used when acessing the class labels
        classes (str): the class labels of the dataset to use, i.e smooth-or-featured-gz2_smooth_fraction, spiral-arm-count-gz2_2_fraction ect
        binarised_labels (bool): whether to binarise the labels
        input_shape (tuple): the input shape of the data, x, y
        input_dim (int): the input dimension of the data, i.e 3 for RGB image
        seed (int): the seed for the random number generator
        ds_fraction (float): the fraction of the dataset to use, range: [0, 1]
        """
        # RNG
        self.seed = seed
        self.numpyRNG = np.random.default_rng(seed)
        
        # File paths
        self.dataset_root = dataset_root
        self.train_parquet_file_path = os.path.join(self.dataset_root, train_catalog)
        self.test_parquet_file_path = os.path.join(self.dataset_root, test_catalog)
        
        # Class labels to use
        self.keys = get_keys(classes)
        self.question_dict = class_labels_to_question()
        self.group_sizes = self.__init_group_sizes()
        
        # Data properties
        self.input_shape = input_shape
        self.input_dim = input_dim
        
        # Dataset properties
        self.binarised_labels = binarised_labels
        
        if ds_fraction <= 0 or ds_fraction > 1:
            raise ValueError("\033[31mThe fraction of the dataset to use must be in the range (0, 1] \033[0m")
        else:
            self.ds_fraction = ds_fraction
        
        # Initialise the datasets
        self.train_dataset = self.__init_dataset(self.train_parquet_file_path)
        self.test_dataset = self.__init_dataset(self.test_parquet_file_path)
        
    #--------------------------------------------------------------------- DATASET(S) INITIALIZATION ---------------------------------------------------------------------------------------
    def __init_dataset(self, catalog_path):
        """
        Initialise a dataset, containing paths to the image files and the labels.
        
        args:
        catalog_path (str): the path to the catalog file
        
        returns:
        dataset (tf.data.Dataset): a zipped dataset containing a tuple of a tf.data.Dataset of image paths and a tf.data.Dataset of labels 
        """ 
        
        print(f"\033[94mInitialising dataset [{catalog_path}]... \033[0m")
        pandas_df = pd.read_parquet(catalog_path)
        pandas_df = pandas_df.fillna(0)
        
        pandas_df['file_loc'] = np.empty(len(pandas_df['id_str']), dtype = str) 
        
        for idx, _ in enumerate(pandas_df['id_str']):  
            pandas_df.loc[idx, 'file_loc'] = os.path.join(self.dataset_root, 'images/', str(pandas_df['subfolder'][idx]) + '/', str(pandas_df['filename'][idx]))

            if not os.path.exists(pandas_df['file_loc'][idx]): 
                pandas_df = pandas_df.drop(idx)
                
        image_path_ds = tf.data.Dataset.from_tensor_slices(pandas_df['file_loc'])
        
        labels = pandas_df[self.keys]
        
        if self.binarised_labels:
            
            print("\033[94mBinarising labels... (this may take a while) \033[0m")
            labels = labels.apply(self.__to_binary, axis = 1, result_type = 'broadcast') 
            print("\033[32mLabels binarised. \033[0m")
            
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        
        print("\033[32mDataset initialised. \033[0m")
        
        return tf.data.Dataset.zip((image_path_ds, labels_ds)).take(int(len(labels_ds) * self.ds_fraction)) # take a fraction of the dataset if you want to use a smaller dataset
    
    def __init_group_sizes(self):
        """
        Initialise the group sizes of the class labels to be used for label binarisation.
        
        returns:
        group_sizes (np.array): the group sizes of the class labels
        """
        values = np.zeros(10, dtype=int) # there are 10 unique questions in the decision tree
        
        for key in self.keys:
            values[self.question_dict[key]] += 1
            
        group_sizes = values[values != 0]
        print(f"Group sizes: {group_sizes}")
        
        return group_sizes
        
    def __to_binary(self, label):
        """
        Split each label vector into each of its constituent questions, i.e. Q1, Q2,..
        for each question convert the answer to binary, i.e. 0.1,0.1,0.8 -> 0,0,1
        
        args:
        label (pd.Series): the label to be binarised
        
        returns:
        new_label (pd.Series): the binarised label
        """
        new_label = np.zeros(len(label))
        offset = 0
        
        for size in self.group_sizes:
            
            group = label[offset:offset + size]
            max_indices = np.flatnonzero(group == np.max(group)) 
            max_idx = self.numpyRNG.choice(max_indices) 
            new_label[max_idx + offset] = 1
            
            offset += size
        
        return pd.Series(new_label)
                
    #--------------------------------------------------------------------------------- IMAGE PROCESSING -------------------------------------------------------------------------------------
    def fetch_image_label_pair(self, image_path, label, architecture, augmentation):
        """
        Fetches an image and its corresponding label from the dataset.
        
        args:
        image_path (tf.Tensor): the path to the image file
        label (tf.Tensor): the label of the image
        architecture (str): the architecture of the model to be used
        augmentation (bool): whether to apply image augmentation
        
        returns:
        (tuple): of the image and its corresponding label
            image (tf.Tensor): the requested image
            label (tf.Tensor): the requested label
        """ 
        image_path = tf.squeeze(image_path)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels = self.input_dim)
        image = tf.image.convert_image_dtype(image, tf.float32) # Convert image dtype from tf.uint8 to tf.float32 when passing to the model
        image = tf.image.resize(image, self.input_shape)
        
        if augmentation:
            augs = tf.random.shuffle(['None', 'flip-lr', 'flip-ud', 'rot90'])[:1]
            for a in augs:
                if a == 'None':
                    image = image
                elif a == 'flip-lr':
                    image = tf.image.random_flip_left_right(image)
                elif a == 'flip-ud':
                    image = tf.image.random_flip_up_down(image)
                elif a == 'rot90':
                    image = tf.image.rot90(image, k = tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32))
        
        # Pre trained models require the input to be preprocessed before passing to the model
        if architecture == "ResNet50":
            image = tf.keras.applications.resnet.preprocess_input(image)
        elif architecture == "ResNet50V2":
            image = tf.keras.applications.resnet_v2.preprocess_input(image)
        elif architecture == "VGG16":
            image = tf.keras.applications.vgg16.preprocess_input(image)
        elif architecture == "VGG19":
            image = tf.keras.applications.vgg19.preprocess_input(image)
        elif architecture == "MobileNetV3Small":
            image = tf.keras.applications.MobileNetV3Small.preprocess_input(image)
        elif architecture == "MobileNetV3Large":
            image = tf.keras.applications.MobileNetV3Large.preprocess_input(image)
        
        return image, label
    
    #TODO: implement image augmentation methods here, i.e rotation, flipping, scaling, translation, brightness, contrast, saturation, hue, denoising, adding noise, reducing resolution, blurring
    #----------------------------------------------------------------------------- VALIDATION DATA SPLITTING ---------------------------------------------------------------------------------
    def get_train_val_split(self, data, val_fraction):
        """
        Split the training dataset into a training and validation dataset.
        
        args:
        data (tf.data.Dataset): the input dataset to be split
        val_fraction (float): the fraction of the dataset to be used for validation, range: [0, 1]
        
        returns: 
        (tuple): of training and validation datasets
            train_split (tf.data.Dataset): the training dataset
            validation_split (tf.data.Dataset): the validation dataset
        """ 
        if val_fraction < 0 or val_fraction > 1:
            raise ValueError("\033[31mThe fraction of the dataset to use must be in the range [0, 1] \033[0m")
        else:
            data = data.shuffle(buffer_size = len(data))  
            train_size = int(len(data) * (1 - val_fraction))
            
        return data.take(train_size), data.skip(train_size)
    
    def get_kfolds(self, data, k):
        """
        Split the training dataset into k-1 folds for training and 1 fold for validation
        to perform k-fold cross validation.
        
        args:
        data (tf.data.Dataset): the input dataset to be split
        k (int): the number of folds to split the dataset into
        returns:
        """
        data = data.shuffle(buffer_size = len(data))
            
        folds = []
        
        for split in range(k):
            validation_data = data.skip(split * (len(data) // k)).take(len(data) // k)
            training_data = data.take(split * (len(data) // k)).concatenate(data.skip((split + 1) * (len(data) // k)))
            folds.append((training_data, validation_data))
        
        return folds
