from packages.imports import os, datetime, tf, keras, np, MaxPooling2D, MaxPooling3D, Conv2D, Conv3D, Sequential, Flatten, Dense, Dropout, KFold, StratifiedKFold, ResNet50, ResNet50V2, VGG16, VGG19, MobileNetV3Small, MobileNetV3Large, Model, csv
from dataframe.keys import get_keys
from programstate import global_state_dict, write_state
# TODO: need to replace last layer of the pre trained model with a dense layer with the number of classes as the output
class ConvolutionalNeuralNetwork():        
    def __init__(self, display_summary, architecture, learning_rate, weight_reg, representation, input_shape, input_dim, loss_function, log_path, checkpoint_path, save_step, saved_weights_path, global_epoch, global_batch, steps_per_epoch):
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
        loss (str): the loss function to use
        log_dir (str): the path to the directory to save the tensorboard logs
        
        """
        # save/load parameters
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.save_step = save_step 
        self.metrics = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.tensorboard_writer = tf.summary.create_file_writer(logdir = self.log_path) 
        
        # network parameters
        self.display_summary = display_summary
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.weight_reg = weight_reg
        self.loss_function = loss_function
        
        # data parameters
        self.label_length = len(get_keys(representation))
        self.input_x, self.input_y = input_shape
        self.input_dim = input_dim
        
        # network
        self.saved_weights_path = saved_weights_path
        self.global_train_batch = global_batch
        self.global_epoch = global_epoch
        self.steps_per_epoch = steps_per_epoch
        self.network = self.__init_network()
    #------------------------------------------------------------------------- NETWORK INITIALIZATION -----------------------------------------------------------------------    
    def __init_network(self) -> tf.keras.Model:
        """
        Initalise a Convolutional Neural Network of a specified architecture using the Keras Sequential API.
        
        returns:
        network (tf.keras.Model): the Convolutional Neural Network
        """
        # Weight regularization
        if self.weight_reg is None:
            wr = None
        else:
            wr = tf.keras.regularizers.l2(self.weight_reg)
        
        match self.architecture: 
            case '128x64xn':
                network = Sequential()
                network.add(Conv2D(filters = 32, kernel_size = (7, 7), strides = 3, activation = 'relu', kernel_regularizer = wr, input_shape = (self.input_x, self.input_y, self.input_dim), name = "Convolutional_1", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(MaxPooling2D(pool_size = (2,2)))
                network.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, activation = 'relu', kernel_regularizer = wr, name = "Convolutional_2", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(MaxPooling2D(pool_size = (2,2)))
                network.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, activation = 'relu', kernel_regularizer = wr, name = "Convolutional_3", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Flatten())
                network.add(Dense(units = 128, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dropout(0.5))
                network.add(Dense(units = 64, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dropout(0.5))
                network.add(Dense(self.label_length, activation = 'softmax', kernel_regularizer = wr, kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())) 
        
                self.__compile_network(network) 
            
            case '':
                network = Sequential()
                network.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (self.input_x, self.input_y, self.input_dim))) 
                
            case 'ResNet50':
                ResNet50_model = ResNet50(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                ResNet50_model = Model(inputs = ResNet50_model.input, outputs = ResNet50_model.get_layer('max_pool').output)
                
                if self.display_summary:
                    ResNet50_model.summary()
                    
                for layer in ResNet50_model.layers:
                        layer.trainable = True 
                 
                network = Sequential()       
                network.add(ResNet50_model)
                network.add(Dense(1000, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dense(self.label_length, activation = 'sigmoid', kernel_regularizer = wr, name = "Output", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.build(input_shape = (None, self.input_x, self.input_y, self.input_dim))
                 
                self.__compile_network(network)    
            
            case 'ResNet50V2':
                ResNet50V2_model = ResNet50V2(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                ResNet50V2_model = Model(inputs = ResNet50V2_model.input, outputs = ResNet50V2_model.get_layer('max_pool').output)
                
                for layer in ResNet50V2_model.layers:
                    layer.trainable = True
                
                if self.display_summary:
                    ResNet50V2_model.summary()
                    
                network = Sequential()
                network.add(ResNet50V2_model)
                network.add(Dense(1000, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dense(self.label_length, activation = 'sigmoid', kernel_regularizer = wr, name = "Output", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.build(input_shape = (None, self.input_x, self.input_y, self.input_dim))
                 
                self.__compile_network(network)
                
            case 'VGG16':
                VGG16_model = VGG16(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                VGG16_model = Model(inputs = VGG16_model.input, outputs = VGG16_model.get_layer('global_max_pooling2d').output)
                
                for layer in VGG16_model.layers:
                    layer.trainable = True
                
                if self.display_summary:
                    VGG16_model.summary()
                    
                network = Sequential()
                network.add(VGG16_model)
                network.add(Dense(500, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dense(500, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dense(self.label_length, activation = 'sigmoid', kernel_regularizer = wr, name = "Output", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.build(input_shape = (None, self.input_x, self.input_y, self.input_dim))
                
                self.__compile_network(network)
                    
            case 'VGG19':
                VGG19_model = VGG19(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                VGG19_model = Model(inputs = VGG19_model.input, outputs = VGG19_model.get_layer('global_max_pooling2d').output)
                
                for layer in VGG19_model.layers:
                    layer.trainable = True
                
                if self.display_summary:
                    VGG19_model.summary()
                    
                network = Sequential()
                network.add(VGG19_model)
                network.add(Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dense(self.label_length, activation = 'sigmoid', kernel_regularizer = wr, name = "Output", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.build(input_shape = (None, self.input_x, self.input_y, self.input_dim))
                
                self.__compile_network(network)
            
            case 'MobileNetV3Small':
                    MobileNetV3Small_model = MobileNetV3Small(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                    MobileNetV3Small_model = Model(inputs = MobileNetV3Small_model.input, outputs = MobileNetV3Small_model.get_layer('max_pool').output)
                    
                    for layer in MobileNetV3Small_model.layers:
                        layer.trainable = True
                    
                    if self.display_summary:
                        MobileNetV3Small_model.summary()
                        
                    network = Sequential()
                    network.add(MobileNetV3Small_model)
                    network.add(Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                    network.add(Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                    network.add(Dense(self.label_length, activation = 'sigmoid', kernel_regularizer = wr, name = "Output", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                    network.build(input_shape = (None, self.input_x, self.input_y, self.input_dim))
                   
                    self.__compile_network(network)
                    
            case 'MobileNetV3Large':
                MobileNetV3Large_model = MobileNetV3Large(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                MobileNetV3Large_model = Model(inputs = MobileNetV3Large_model.input, outputs = MobileNetV3Large_model.get_layer('max_pool').output)
                
                for layer in MobileNetV3Large_model.layers:
                    layer.trainable = True
                
                if self.display_summary:
                    MobileNetV3Large_model.summary()
                
                network = Sequential()
                network.add(MobileNetV3Large_model)
                network.add(Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.add(Dense(self.label_length, activation = 'sigmoid', kernel_regularizer = wr, name = "Output", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
                network.build(input_shape = (None, self.input_x, self.input_y, self.input_dim))
                    
                self.__compile_network(network)
                
            case _:
                raise ValueError(f"\033[31mInvalid architecture parameter provided: {self.architecture} \033[31m\033[0m")
        
        if (self.display_summary):
            network.summary()
            
        return network   
    
    def __compile_network(self, network):
        """
        Compiles the optimiser of a network.
        If the network is being loaded from a checkpoint, we overwrite the randomly initialised weights with the one from the save.
        
        args:
        network (tf.keras.Model): the network to compile
        """
        if self.saved_weights_path:
            self.load(network)
            print(f"\033[32mLoaded save state [{self.saved_weights_path}]! \033[0m")
        network.compile(loss = self.loss_function, optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, epsilon = 1e-07), metrics = ['accuracy'])
            
    #------------------------------------------------------------------------- NETWORK TRAINING AND TESTING ---------------------------------------------------------------------------
    def train(self, train_data, validation_data, dataframe, train_batch_size, val_batch_size, val_steps, epochs, augmentation):
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
        augmentation (bool): whether to augment the data
        """
        
        train_data = train_data.map(
            lambda image, label: dataframe.fetch_image_label_pair(image, label, self.architecture, augmentation), 
            num_parallel_calls = tf.data.experimental.AUTOTUNE 
        ).shuffle(
            buffer_size = 256, reshuffle_each_iteration = True
        ).batch(
            train_batch_size
        ).repeat(
        ).prefetch(
            tf.data.experimental.AUTOTUNE 
        )
        
        validation_data = validation_data.map(
            lambda image, label: dataframe.fetch_image_label_pair(image, label, self.architecture, augmentation),
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        ).shuffle(
            buffer_size = 256, reshuffle_each_iteration = True
        ).batch(
            val_batch_size
        ).repeat(
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        ).repeat()
        
        # this callback can't log the history of parameters per iteration, but it has other useful metrics which is why I'm using it and the customcallback class.
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir = self.log_path,
            histogram_freq = 1,
            write_graph = True,
            write_images = True,
            write_steps_per_second = True,
            update_freq = 'epoch',
        )
        
        self.network.fit(
            train_data,
            epochs = epochs,
            verbose = 1,
            callbacks = [ 
                CustomCallbacks(self), 
                tensorboard_callback
            ], 
            validation_data = validation_data,
            shuffle = True,
            initial_epoch = self.global_epoch,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = val_steps,
            validation_batch_size = val_batch_size,
            validation_freq = 1,
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
            lambda image, label: dataframe.fetch_image_label_pair(image, label, self.architecture, self.augmentation),
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        ).batch(
            test_batch_size
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        
        self.network.evaluate(
            test_data,
            batch_size = test_batch_size,
            verbose = 1,
            steps = test_steps,
            callbacks = None,
            return_dict = True
        )
        
    #----------------------------------------------------------------- NETWORK & METRICS SAVING ----------------------------------------------------------------   
    def save(self, metrics):
        """
        Save the model and write the metrics to a csv file.
        
        args:
        metrics (list): the metrics to write to the csv file
        """
        self.__save_model()
        
        for metric in metrics:
            self.__write_metrics_to_csv(metric_keys = [metric])
            self.metrics[metric] = []
    
    def __save_model(self):
        """
        Save the network to a file.
        Saved as: [checkpoint_path]/checkpoint-[timestamp]/.weights.h5.
        """
        save_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        rootpath = os.path.join(self.checkpoint_path, "checkpoint-" + f"{save_time}")
        os.makedirs(rootpath, exist_ok = True)
        savepath = os.path.join(rootpath, 'model.weights.h5')
        
        global_state_dict['saved_weights_path'] = savepath
        self.network.save_weights(savepath)
        
        self.__write_parameters_to_json(rootpath)
        
    def __write_parameters_to_json(self, path):
        """
        Write the parameters passed to the main() function in main.py to a json file along with the global_train_batch and 
        epoch the model was saved on.
        """
        global_state_dict['global_epoch'] = self.global_epoch
        global_state_dict['global_batch'] = self.global_train_batch 
        
        write_state(path)
         
    def __write_metrics_to_csv(self, metric_keys):
        """
        Write the metrics to a csv file.
        
        args:
        metric_keys (list): the metrics to write to the csv file
        """
        for metric in metric_keys:
            filename = os.path.join(self.log_path, f"{metric}.csv")
            
            if not os.path.exists(filename):
                os.makedirs(os.path.dirname(filename), exist_ok = True)
                
            with open(filename, 'a', newline='') as writer:
                writer = csv.writer(writer)
                for value in self.metrics[metric]:
                    writer.writerow([value])
    
    #----------------------------------------------------------------- NETWORK LOADING ----------------------------------------------------------------
    def load(self, network):
        """
        load weights and biases of the network from a file and the training state of the network from a file.
        
        network (tf.keras.Model): the network to load the weights and biases into
        """
        
        print(f"\033[32mLoaded training state: global_epoch = {self.global_epoch}, global_train_batch = {self.global_train_batch} \033[0m")
        weight_path = os.path.join(self.saved_weights_path)
        
        print(f"\033[33mLoading weights and biases from [{weight_path}]... \033[0m")
        network.load_weights(weight_path)  
        print(f"\033[32mLoaded weights and biases! \033[0m")
        
class CustomCallbacks(keras.callbacks.Callback): # Inherits from keras.callbacks.Callback
        def __init__(self, CNNreference):
            """
            Construct a CustomCallbacks object to allow for custom callbacks during training,
            such as saving the model and writing metrics to a tensorboard log file.
            
            args:
            CNNreference (ConvolutionalNeuralNetwork): the ConvolutionalNeuralNetwork object
            """
            super(CustomCallbacks,self).__init__() # calls the constructor of the parent class 
            self.CNNreference = CNNreference
            
        def on_batch_end(self, batch, logs = None): 
            """
            Save the metrics to the tensorboard log file every batch.
            
            args:
            batch (int): the batch number
            logs (dict): the logs to save
            """
            with self.CNNreference.tensorboard_writer.as_default():
                # metrics per batch
                self.CNNreference.metrics['loss'].append(logs.get('loss'))
                self.CNNreference.metrics['accuracy'].append(logs.get('accuracy'))
                
                tf.summary.scalar(name = 'Loss', data = logs.get('loss'), step = self.CNNreference.global_train_batch)
                tf.summary.scalar(name = 'Accuracy', data = logs.get('accuracy'), step = self.CNNreference.global_train_batch)
                self.CNNreference.tensorboard_writer.flush()
                
                # save the model and metrics every save_step batches
                if self.CNNreference.global_train_batch % self.CNNreference.save_step == 0 and self.CNNreference.global_train_batch != 0 and self.CNNreference.global_train_batch % self.CNNreference.steps_per_epoch != 0: # dont save when the save step is at the end or start of the epoch
                    
                    self.CNNreference.save(metrics = ['loss', 'accuracy'])
                    
                self.CNNreference.global_train_batch += 1
                
        def on_epoch_end(self, epoch, logs = None): 
            """
            Save the metrics to the tensorboard log file every epoch.
            
            args:
            epoch (int): the epoch number
            logs (dict): the logs to save
            """
            with self.CNNreference.tensorboard_writer.as_default():
                # metrics per epoch
                self.CNNreference.metrics['val_loss'].append(logs.get('val_loss'))
                self.CNNreference.metrics['val_accuracy'].append(logs.get('val_accuracy'))
                
                tf.summary.scalar(name = 'Validation loss', data = logs.get('val_loss'), step = epoch)
                tf.summary.scalar(name = 'Validation accuracy', data = logs.get('val_accuracy'), step = epoch)
                self.CNNreference.tensorboard_writer.flush()
                
                # save the model and metrics at the end of the epoch
                self.CNNreference.save(metrics = ['val_loss', 'val_accuracy'])
                self.CNNreference.global_epoch += 1