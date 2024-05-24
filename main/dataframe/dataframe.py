from packages.imports import os, tf, pd, np, cv2
from dataframe.keys import get_keys, class_labels_to_question

def get_paths(root, dataset_root, checkpointroot, logroot, time_stamp, architecture, classes):
    """
    Get the paths to the dataset, checkpoint and log directories.
    
    args:
    root (str): the path to the root directory
    dataset_root (str): the path to the dataset root directory
    checkpointroot (str): the path to the checkpoint root directory
    logroot (str): the path to the log root directory
    time_stamp (str): the date and time to use when naming the paths
    architecture (str): the architecture of the network
    classes (str): the labels used when training the network
    
    returns:
    dataset_root (str): the path to the dataset root directory
    checkpoint_path (str): the path to the checkpoint directory
    log_path (str): the path to the log directory
    """
    if not os.path.exists(os.path.join(root, checkpointroot + f"/{architecture}-{classes}-{time_stamp}")):
        os.makedirs(os.path.join(root, checkpointroot + f"/{architecture}-{classes}-{time_stamp}"))
    
    if not os.path.exists(os.path.join(root, logroot + f"/{architecture}-{classes}-{time_stamp}")):
        os.makedirs(os.path.join(root, logroot + f"/{architecture}-{classes}-{time_stamp}"))
        
    if not os.path.exists(os.path.join(root, 'logs/', 'predictions/')):
        os.makedirs(os.path.join(root, 'logs/', 'predictions/'))
        
    dataset_root = os.path.join(root, dataset_root)
    checkpoint_path = os.path.join(root, checkpointroot + f"/{architecture}-{classes}-{time_stamp}")
    log_path = os.path.join(root, logroot + f"/{architecture}-{classes}-{time_stamp}")
    predictions_path = os.path.join(root, 'logs/', 'predictions/')
    
    return dataset_root, checkpoint_path, log_path, predictions_path

class DataFrame():
    def __init__(self, dataset_root, catalog, classes, build_paths, binarised_labels, input_shape, input_dim, seed, ds_fraction, weight_mode, beta):
        """
        Construct a DataFrame containing the represenation of the training and test data
        
        args:
        dataset_root (str): path to the dataset root directory
        catalog (str): the name of the catalog file
        keys (list): list of the keys to be used when acessing the class labels
        classes (str): the class labels of the dataset to use, i.e smooth-or-featured-gz2_smooth_fraction, spiral-arm-count-gz2_2_fraction ect
        build_paths (bool): whether to build the file paths
        binarised_labels (bool): whether to binarise the labels
        input_shape (tuple): tuple of ints, the input shape of the data, x, y
        input_dim (int): the input dimension of the data, i.e 3 for RGB image
        seed (int): the seed for the random number generator
        ds_fraction (float): the fraction of the dataset to use, range: [0, 1]
        weight_mode (str): the method to use when calculating the class weights, options: 'inverse', 'inverse-sqrt', 'effective_num_samples', None
        """
        # RNG
        self.seed = seed
        self.numpyRNG = np.random.default_rng(seed)
        
        # File paths
        self.dataset_root = dataset_root
        self.catalog = catalog
        self.catalog_path = os.path.join(self.dataset_root, self.catalog)
        
        # Class labels to use
        self.keys = get_keys(classes)
        self.question_dict = class_labels_to_question()
        self.group_sizes = self.__init_group_sizes()
        self.weight_mode = weight_mode
        self.beta = beta
        self.class_weights = None
        
        # Data properties
        self.input_shape = input_shape
        self.input_dim = input_dim
        
        # Dataset properties
        self.build_paths = build_paths
        self.binarised_labels = binarised_labels
        
        if ds_fraction <= 0 or ds_fraction > 1:
            raise ValueError("\033[31mThe fraction of the dataset to use must be in the range (0, 1] \033[0m")
        else:
            self.ds_fraction = ds_fraction
        
        # Initialise the dataset
        self.dataset = self.__init_dataset(self.catalog_path)
        
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
        
        if self.build_paths: 
            print("\033[94mBuilding file paths... (This may take a while)\033[0m")
            pandas_df['file_loc'] = pandas_df.apply(self.__build_file_path, axis = 1)
            pandas_df.dropna(subset = ['file_loc'], inplace = True)
               
        image_path_ds = tf.data.Dataset.from_tensor_slices(pandas_df['file_loc'])
        
        labels = pandas_df[self.keys]
        
        if self.binarised_labels:
            
            print("\033[94mBinarising labels... (this may take a while) \033[0m")
            labels = labels.apply(self.__to_binary, axis = 1, result_type = 'broadcast') 
            print("\033[32mLabels binarised. \033[0m")
        
            self.class_weights = self.__create_weights(pandas_dataframe = labels)
            print(f"\033[33mloss function weights: {self.class_weights}\033[0m")
        else:
            self.class_weights = {idx: 1 for idx, _ in enumerate(self.keys)}
             
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        
        print("\033[32mDataset initialised. \033[0m")
        
        return tf.data.Dataset.zip((image_path_ds, labels_ds)).take(int(len(labels_ds) * self.ds_fraction)) # take a fraction of the dataset if you want to use a smaller dataset
    
    def __build_file_path(self, row):
        """
        Construct the file path to the image file.
        
        args:
        row (pd.Series): the row of the dataframe
        
        returns:
        file_loc (str): the path to the image file if it exists, else None
        """
        file_loc = os.path.join(self.dataset_root, 'images/', str(row['subfolder']), str(row['filename']))
        return file_loc if os.path.exists(file_loc) else None

    def __init_group_sizes(self):
        """
        Initialise the group sizes of the class labels to be used for label binarisation.
        
        returns:
        group_sizes (np.array): the group sizes of the class labels
        """
        values = np.zeros(11, dtype=int) # there are 11 unique questions in the decision tree
        
        for key in self.keys:
            values[self.question_dict[key]] += 1
            
        group_sizes = values[values != 0]
        print(f"\033[33mGroup sizes: {group_sizes}\033[0m")
        
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
            
            if np.max(group) < 0.5 and size == 1: # if the max value is less than 0.5 and there's only 1 label in the group it should be 0 othewise values like 0.1 -> 1 which is dumb.
                new_label[max_idx + offset] = 0
            else:
                new_label[max_idx + offset] = 1 
            
            offset += size
        
        return pd.Series(new_label)
                
    def get_feature_weights(self):
        """
        Get a dictionary containing the weights to be used in the loss function for each class label.
        
        returns:
        weight_dict (dict): a dictionary mapping of class indices (integers) to a weight (float)
        """
        return self.class_weights
    
    def __create_weights(self, pandas_dataframe):
        """
        create class weights for the dataset by finding the fraction of galaxies that have a specific feature (= 1 has to be binary encoded)
        
        args:
        pandas_dataframe (pd.DataFrame): the dataframe containing the class labels
        
        returns:
        weights (list): a list of the weights for each class label
        """       
        weights = []
        
        if self.beta is None: # calculate beta based off the formula beta = (N - 1) / N (4. Class-Balanced Loss)
            self.beta = (len(pandas_dataframe) - 1) / len(pandas_dataframe)
        
        for key in self.keys:
            num = (pandas_dataframe[key] == 1).sum()
            fraction = num / len(pandas_dataframe[key])
            
            if self.weight_mode == 'inverse':
                weight = 1 / fraction
                
            elif self.weight_mode == 'inverse-sqrt':
                weight = 1 / np.sqrt(fraction)
            
            # https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf 
            # 4. Class-Balanced Loss:
            elif self.weight_mode == 'effective_num_samples': # beta = 0 -> no reweighting. beta = 1 -> reweighting by inverse class frequency, 
                effective_num = (1 - self.beta**num) / (1 - self.beta)
                weight = - 1 / effective_num # 4.2. Class-Balanced Sigmoid Cross-Entropy Loss: CB = -1/weights * loss_sigmoid_cross_entropy(z, y)

            elif self.weight_mode is None:  
                weight = 1
                
            else:
                raise ValueError(f"\033[31mInvalid weight mode!: {self.weight_mode}\033[0m")
                
            weights.append(weight)
            
        if self.weight_mode == 'effective_num_samples': # 4. Class-Balanced Loss: sum of weights = num classes (in paper 1/E_n used to represent this normalised weighting factor for convinence)
            total = np.sum(weights)
            weights = [(weight / total) * len(self.keys) for weight in weights]
             
        return {idx: weight for idx, weight in enumerate(weights)}           
        
    #--------------------------------------------------------------------------------- IMAGE PROCESSING -------------------------------------------------------------------------------------
    def fetch_image_label_pair(self, image_path, label, architecture, augmentation, crop, crop_size):
        """
        Fetches an image and its corresponding label from the dataset.
        
        args:
        image_path (tf.Tensor): the path to the image file
        label (tf.Tensor): the label of the image
        architecture (str): the architecture of the model 
        augmentation (bool): whether to apply image augmentation
        crop (bool): whether to crop the image to input_shape around the center of the image
        crop_size (tuple): the dimensions to crop the image to
            
        returns:
        (tuple): of the image and its corresponding label
            image (tf.Tensor): the requested image
            label (tf.Tensor): the requested label
        """ 
      
        image_path = tf.squeeze(image_path)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels = self.input_dim)
        
        if crop:
            image = self.__crop_image(image, crop_size)
        else:
            image = tf.image.resize(image, self.input_shape) 
            
        image = self.__pretrained_preprocess(image, architecture) # Pre trained models require the input to be preprocessed before passing to the model 
            
        if augmentation:
            augs = tf.random.shuffle(['None', 'flip-lr', 'flip-ud', 'rot90'])[:1]
            image = self.__augment_image(image, augs)
        
        label = tf.cast(label, tf.float32)
            
        return image, label
    
    def __crop_image(self, image, crop_size):
        """
        Crop an image around it's center to a specified size
        
        args:
        image (tf.Tensor): the image to crop
        crop_size (tuple): the x and y dimensions of the desired crop
        
        returns
        image (tf.Tensor): the cropped image
        """
        center = tf.shape(image)[:2] // 2
        crop = tf.constant(crop_size)
        start = center - crop // 2
        start = tf.maximum(start, 0)
        return tf.image.crop_to_bounding_box(image, start[0], start[1], crop[0], crop[1])
    
    @tf.function
    def __augment_image(self, image, aug):
        """
        Apply a random augmentation to an image
        
        args:
        image (tf.Tensor): The image to augment
        aug (str): The augmentation to apply
        
        returns:
        image (tf.Tensor): The augmented image
        """
        if aug == 'None':
            image = image
        elif aug == 'flip-lr':
            image = tf.image.random_flip_left_right(image)
        elif aug == 'flip-ud':
            image = tf.image.random_flip_up_down(image)
        elif aug == 'rot90':
            image = tf.image.rot90(image, k = tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32))
            
        return image
    
    def __pretrained_preprocess(self, image, architecture):
        """
        Preprocess the image before passing it to a pretrained model
        
        args:
        image (tf.Tensor): the image to preprocess
        architecture (str): the architecture of the model 
        
        returns:
        image (tf.Tensor): the processed image
        """
        if architecture == "ResNet50":
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.resnet.preprocess_input(image)
        elif architecture == "ResNet50V2":
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.resnet_v2.preprocess_input(image)
        elif architecture == "VGG16":
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.vgg16.preprocess_input(image)
        elif architecture == "VGG19":
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.vgg19.preprocess_input(image)
        elif architecture == "MobileNetV3Small":
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
        elif architecture == "MobileNetV3Large":
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
        elif architecture == "InceptionV3":
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.inception_v3.preprocess_input(image)
        elif architecture == "Xception":
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.xception.preprocess_input(image)
        else:
            image = tf.image.convert_image_dtype(image, tf.float32)
        return image  
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
    
    def get_kfolds(self, data, k): # NOTE: not sure if this works properly, might just get rid of this
        """
        Split the training dataset into k-1 folds for training and 1 fold for validation
        to perform k-fold cross validation.
        
        args:
        data (tf.data.Dataset): the input dataset to be split
        k (int): the number of folds to split the dataset into
        returns:
        
        folds (list): a list of tuples containing the training and validation datasets
        """
        data = data.shuffle(buffer_size = len(data))
            
        folds = []
        
        for split in range(k):
            validation_data = data.skip(split * (len(data) // k)).take(len(data) // k)
            training_data = data.take(split * (len(data) // k)).concatenate(data.skip((split + 1) * (len(data) // k)))
            folds.append((training_data, validation_data))
        
        return folds

    def show_image_label_pair(self, root, idx, architecture, augmentation, crop, crop_size):
        """
        Save an image-label pair from the function fetch_image_label_pair to a file. 
        (useful for testing)
        
        args:
        root (str): the root directory
        idx (int): the image-label pair to save
        architecture (str): the architecture of the network (required by fetch_image_label_pair)
        augmentation (bool): whether to augment the images (required by fetch_image_label_pair)
        crop (bool): whether to crop the image (required by fetch_image_label_pair)
        crop_size (tuple): tuple of ints, the dimensions to crop the image to (required by fetch_image_label_pair)
        """
        log_path = os.path.join(root, 'logs/image-label-pairs')
            
        if not os.path.exists(log_path):
            os.makedirs(log_path)
                
        for element in self.dataset.skip(idx).take(1):
            
            im_path = os.path.join(log_path, f'image-{idx}.png')
            la_path = os.path.join(log_path, f'label-{idx}.txt')
            
            image_path, label = element
            image, label = self.fetch_image_label_pair(image_path, label, architecture, augmentation, crop, crop_size)
            
            image = cv2.cvtColor(image.numpy()*255, cv2.COLOR_RGB2BGR)
            cv2.imwrite(im_path, image)

            with open(la_path, 'w') as file:
                file.write(str(label))     
