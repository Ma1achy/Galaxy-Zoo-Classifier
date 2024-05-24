from packages.imports import os, datetime, tf, keras, np, MaxPooling2D, MaxPooling3D, Conv2D, Conv3D, Sequential, Flatten, Dense, Dropout, LayerNormalization, MultiHeadAttention, KFold, StratifiedKFold, Xception, ResNet50, ResNet50V2, VGG16, VGG19, MobileNetV3Small, MobileNetV3Large, InceptionV3, Model, csv, json  # noqa: F401
from dataframe.keys import get_keys
from programstate import global_state_dict, write_state
from keras import layers
from network.transformer import mlp, augmentation_layers, Patches, PatchEncoder, SeparableConvMultiHeadAttention, AddCLSToken, ConvolutionalEmbedding
from network.CTT import ConvolutionalTokeniser, PositionEmbedding, SequencePooling, StochasticDepth
from network.CvT import MLP
class Network():        
    def __init__(self, display_summary, architecture, learning_rate, weight_reg, activation_function, representation, input_shape, input_dim, loss_function, num_patches, projection_dim, num_heads, transformer_layers, mlp_head_units, log_path, checkpoint_path, save_step, saved_weights_path, global_epoch, global_batch, steps_per_epoch, weighting, predictions_path):
        """
        Construct a Neural Network of a specified architecture: ViT, ResNet50, ResNet50V2, VGG16, VGG19, MobileNetV3Small, MobileNetV3Large, Xception.
        
        args:
        display_summary (bool): whether to display the summary of the network
        architecture (str): the architecture of the network
        learning_rate (float): the learning rate of the network
        weight_reg (float): the weight regularisation of the network
        activation_function (str): the activation function of the network
        representation (str): shorthand for the class labels used
        input_shape (tuple): tuple of ints, the shape of the input data
        input_dim (int): the number of dimensions of the input data
        loss_function (str): the loss function of the network
        num_patches (int): the number of patches to split the image into
        projection_dim (int): the projection dimension of the transformer model
        num_heads (int): the number of heads in the transformer model
        transformer_layers (int): the number of transformer layers
        mlp_head_units (list): list of ints, the hidden units of the mlp head
        log_path (str): the path to save the tensorboard logs
        checkpoint_path (str): the path to save the model
        save_step (int): the number of batches to save the model
        saved_weights_path (str): the path to save the weights
        global_epoch (int): the epoch the model was saved on
        global_batch (int): the batch the model was saved on
        steps_per_epoch (int): the number of steps per epoch
        weighting (dict): the weighting of the class labels
        predictions_path (str): the path to save the predictions
        """
        # save/load parameters
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.save_step = save_step 
        self.metrics = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'root_mean_squared_error': [], 'mean_absolute_error': [], 'binary_accuracy': []}
        self.tensorboard_writer = tf.summary.create_file_writer(logdir = self.log_path) 
        
        # network parameters
        self.display_summary = display_summary
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.weight_reg = weight_reg
        self.activation_function = activation_function
        self.loss_function = loss_function
        
        # transformer model hyperparameters
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.positional_embedding = True
        
        # data parameters
        self.representation = representation
        self.label_length = len(get_keys(representation))
        self.input_x, self.input_y = input_shape
        self.input_dim = input_dim
        
        if weighting is None:
            self.weighting = {i: 1 for i in range(self.label_length)}
        else:
            self.weighting = weighting
        
        # network
        self.saved_weights_path = saved_weights_path
        self.global_train_batch = global_batch
        self.global_epoch = global_epoch
        self.steps_per_epoch = steps_per_epoch

        self.network = self.__init_network()
        
        # model testing
        self.predictions = None
        self.global_evaluate_batch = 0
        self.test_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.predictions_path = predictions_path
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
            case 'ViT':
                transformer_units = [
                    self.projection_dim * 2, 
                    self.projection_dim,
                ]
                
                inputs = keras.Input(shape = (self.input_x, self.input_y, self.input_dim))
                augs = augmentation_layers(self.input_x, self.input_y)
                augs = augs(inputs)
                patches = Patches(self.input_x, self.num_patches)(augs)
                encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

                for _ in range(self.transformer_layers):
                    x1 = LayerNormalization(epsilon = 1e-6)(encoded_patches) # layer normalisation 1
                    attention_output = MultiHeadAttention(num_heads = self.num_heads, key_dim = self.projection_dim, dropout = 0.1)(x1, x1) 
                    
                    # skip connection 1
                    x2 = layers.Add()([attention_output, encoded_patches])
                    x3 = LayerNormalization(epsilon = 1e-6)(x2) # layer normalisation 2
                    
                    x3 = mlp(x3, hidden_units = transformer_units, dropout_rate = 0.1)
                    
                    # skip connection 2
                    encoded_patches = layers.Add()([x3, x2])
                    
                # create a [batch_size, projection_dim] tensor.
                representation = LayerNormalization(epsilon = 1e-6)(encoded_patches)
                representation = Flatten()(representation)
                representation = Dropout(0.5)(representation) 
                
                features = mlp(representation, hidden_units = self.mlp_head_units, dropout_rate = 0.5)
                logits = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(features)
                
                network = Model(inputs = inputs, outputs = logits)
                
                self.__compile_network(network) 
                
            case 'CvT':
                units = [
                    self.projection_dim * 2, 
                    self.projection_dim,
                ]
                 
                inputs = keras.Input(shape=(self.input_x, self.input_y, self.input_dim))
                convolutional_tokeniser = ConvolutionalEmbedding(patch_size = 3, embed_dim = self.projection_dim, stride = 1, padding = 'valid', norm_layer = True) 
                encoded_patches = convolutional_tokeniser(inputs)

                TransformerBlock1 = Sequential([
                    LayerNormalization(epsilon = 1e-6),
                    SeparableConvMultiHeadAttention(num_heads = self.num_heads, key_dim = self.projection_dim, dropout = 0.1, kernel_size = 7, stride = 4, padding = 'valid'),
                    LayerNormalization(epsilon = 1e-6),
                    MLP(hidden_units = units, dropout_rate = 0.1),
                ], name ='Transformer_Block_1'
                )
                
                TransformerBlock2 = Sequential([
                    LayerNormalization(epsilon = 1e-6),
                    SeparableConvMultiHeadAttention(num_heads = self.num_heads, key_dim = self.projection_dim, dropout = 0.1, kernel_size = 5, stride = 4, padding = 'valid'),
                    LayerNormalization(epsilon = 1e-6),
                    MLP(hidden_units = units, dropout_rate = 0.1),
                ], name ='Transformer_Block_2'
                )
                
                TransformerBlock3 = Sequential([
                    AddCLSToken(cls_token = tf.zeros(shape=(1, self.projection_dim))),
                    LayerNormalization(epsilon = 1e-6),
                    SeparableConvMultiHeadAttention(num_heads = self.num_heads, key_dim = self.projection_dim, dropout = 0.1, kernel_size = 3, stride = 1, padding = 'valid'),
                    LayerNormalization(epsilon = 1e-6),
                    MLP(hidden_units = units, dropout_rate = 0.1)
                ], name ='Transformer_Block_3'
                )
                
                x = TransformerBlock1(encoded_patches)
                x = TransformerBlock2(x)
                x = TransformerBlock3(x)
          
                representation = LayerNormalization(epsilon=1e-5)(x)
                representation = Flatten()(representation)
                representation = Dropout(0.5)(representation)
                
                features = mlp(x = representation, hidden_units = self.mlp_head_units, dropout_rate = 0.5)
                logits = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(features)
                
                network = Model(inputs = inputs, outputs = logits)
                
                self.__compile_network(network)
                    
            case 'CCT':
                transformer_units = [
                self.projection_dim,
                self.projection_dim,
                ]
                
                stochastic_depth_rate = 0.1
                
                inputs = layers.Input(shape = (self.input_x, self.input_y, self.input_dim))
                
                covolutional_tokeniser = ConvolutionalTokeniser(kernel_size = 3, stride = 1, padding = 1, pooling_kernel_size = 3, pooling_stride = 2, num_conv_layers = 2, num_output_channels = [self.projection_dim, self.projection_dim], positional_embedding = True)
                encoded_patches = covolutional_tokeniser(inputs)
                
                if self.positional_embedding:
                    sequence_length = encoded_patches.shape[1]
                    encoded_patches = PositionEmbedding(sequence_length = sequence_length)(encoded_patches)
                    
                dpr = [x for x in np.linspace(0, stochastic_depth_rate, self.transformer_layers)]  # stochastic depth decay rule
                
                for i in range(self.transformer_layers):
                    x1 = LayerNormalization(epsilon = 1e-6)(encoded_patches) # normalisation layer 1
                    attention_output = MultiHeadAttention(num_heads = self.num_heads, key_dim = self.projection_dim, dropout = 0.1)(x1, x1) # multi-head attention layer
                    
                    # skip connection 1
                    attention_output = StochasticDepth(dpr[i])(attention_output) 
                    x2 = layers.Add()([attention_output, encoded_patches])
                    x3 = LayerNormalization(epsilon = 1e-6)(x2) # normalisation layer 2
                    x3 = mlp(x3, hidden_units = transformer_units, dropout_rate = 0.1)
                    
                    # skip connection 2
                    x3 = StochasticDepth(dpr[i])(x3) 
                    encoded_patches = layers.Add()([x3, x2])
                
                # Apply sequence pooling.
                representation = LayerNormalization(epsilon=1e-5)(encoded_patches)
                weighted_representation = SequencePooling()(representation)
                weighted_representation = mlp(x = weighted_representation, hidden_units = self.mlp_head_units, dropout_rate = 0.5)
                
                logits = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(weighted_representation)
                network = keras.Model(inputs=inputs, outputs=logits)
                
                self.__compile_network(network)
                 
            case 'ResNet50':
                ResNet50_model = ResNet50(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                ResNet50_output = Flatten()(ResNet50_model.output)
                Dense_layer = Dense(1000, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(ResNet50_output)
                Classifier = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(Dense_layer)
                
                network = Model(inputs = ResNet50_model.input, outputs = Classifier)
                
                for layer in network.layers:
                        layer.trainable = True 
                 
                self.__compile_network(network)    
            
            case 'ResNet50V2':
                ResNet50V2_model = ResNet50V2(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                ResNet50V2_output = Flatten()(ResNet50V2_model.output)
                Dense_layer = Dense(1000, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(ResNet50V2_output)
                Classifier = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(Dense_layer)
                
                network = Model(inputs = ResNet50V2_model.input, outputs = Classifier)
                
                for layer in network.layers:
                    layer.trainable = True
                 
                self.__compile_network(network)
                
            case 'VGG16':
                VGG16_model = VGG16(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                VGG16_output = Flatten()(VGG16_model.output)
                Dense_layer_1 = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(VGG16_output)
                Dense_layer_2 = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(Dense_layer_1)
                Classifier = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(Dense_layer_2)
                
                network = Model(inputs = VGG16_model.input, outputs = Classifier)
                
                for layer in network.layers:
                    layer.trainable = True
                
                self.__compile_network(network)
                    
            case 'VGG19':
                VGG19_model = VGG19(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                VGG19_output = Flatten()(VGG19_model.output)
                Dense_layer_1 = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(VGG19_output)
                Dense_layer_2 = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(Dense_layer_1)
                Classifier = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(Dense_layer_2)
                
                network = Model(inputs = VGG19_model.input, outputs = Classifier)
                
                for layer in network.layers:
                    layer.trainable = True
                
                self.__compile_network(network)
            
            case 'MobileNetV3Small':
                    MobileNetV3Small_model = MobileNetV3Small(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max", include_preprocessing = True)
                    MobileNetV3Small_output = Flatten()(MobileNetV3Small_model.output)
                    Dense_layer_1 = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(MobileNetV3Small_output)
                    Dense_layer_2 = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(Dense_layer_1)
                    Classifier = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(Dense_layer_2)
                    
                    network = Model(inputs = MobileNetV3Small_model.input, outputs = Classifier)
                    
                    for layer in network.layers:
                        layer.trainable = True
                
                    self.__compile_network(network)
                    
            case 'MobileNetV3Large':
                MobileNetV3Large_model = MobileNetV3Large(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max", include_preprocessing = True)
                MobileNetV3Large_output = Flatten()(MobileNetV3Large_model.output)
                Dense_layer_1 = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(MobileNetV3Large_output)
                Dense_layer_2 = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_2", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(Dense_layer_1)
                Classifier = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(Dense_layer_2)
                
                network = Model(inputs = MobileNetV3Large_model.input, outputs = Classifier)
                
                for layer in network.layers:
                    layer.trainable = True
                
                self.__compile_network(network)
            
            case 'Xception':
                Xception_model = Xception(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                Xception_output = Flatten()(Xception_model.output)
                Dense_layer = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(Xception_output)
                Classifier = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(Dense_layer)
                
                network = Model(inputs = Xception_model.input, outputs = Classifier)
                
                for layer in network.layers:
                    layer.trainable = True
                
                self.__compile_network(network)
                
            case 'InceptionV3':
                InceptionV3_model = InceptionV3(include_top = False, weights = 'imagenet', input_shape = (self.input_x, self.input_y, self.input_dim), pooling = "max")
                InceptionV3_output = Flatten()(InceptionV3_model.output)
                Dense_layer = Dense(4096, activation = 'relu', kernel_regularizer = wr, name = "Dense_1", kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(InceptionV3_output)
                Classifier = Dense(self.label_length, activation = self.activation_function, kernel_regularizer = wr, name = "Predictions", kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros())(Dense_layer)
                
                network = Model(inputs = InceptionV3_model.input, outputs = Classifier)
                
                for layer in network.layers:
                    layer.trainable = True
                    
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
            
        network.compile(loss = self.loss_function, optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, epsilon = 1e-07), metrics = ['accuracy', 'root_mean_squared_error', 'mean_absolute_error', 'binary_accuracy'])
  
    #------------------------------------------------------------------------- NETWORK TRAINING AND TESTING ---------------------------------------------------------------------------
    def train(self, train_data, validation_data, dataframe, train_batch_size, val_batch_size, val_steps, epochs, augmentation, crop, crop_size):
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
        crop (bool): whether to crop the image to input_shape around the center of the image
        crop_size (tuple): the dimensions to crop the image to
        """
        
        train_data = train_data.shuffle(
            buffer_size = train_data.cardinality(), reshuffle_each_iteration = True
        ).map(
            lambda image, label: dataframe.fetch_image_label_pair(image, label, self.architecture, augmentation, crop, crop_size), 
            num_parallel_calls = tf.data.experimental.AUTOTUNE 
        ).batch(
            train_batch_size
        ).repeat(
        ).prefetch(
            tf.data.experimental.AUTOTUNE 
        )
        
        validation_data = validation_data.shuffle(
            buffer_size = validation_data.cardinality(), reshuffle_each_iteration = True
        ).map(
            lambda image, label: dataframe.fetch_image_label_pair(image, label, self.architecture, augmentation, crop, crop_size),
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        ).batch(
            val_batch_size
        ).repeat(
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        
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
                TrainingCustomCallbacks(self), 
                tensorboard_callback
            ], 
            validation_data = validation_data,
            shuffle = True,
            initial_epoch = self.global_epoch,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = val_steps,
            validation_batch_size = val_batch_size,
            validation_freq = 1,
            class_weight = self.weighting
        )
        
    def evaluate(self, test_data, dataframe, test_batch_size, augmentation, crop, crop_size):
        """
        Evaluate the network on the test data using model.evaluate().
        
        args:
        test_data (tf.data.Dataset): the test dataset
        dataframe (DataFrame): An instance of DataFrame containing the representation of the training and test data
        test_batch_size (int): the batch size of the test data
        """
        
        test_data = test_data.shuffle(
            buffer_size = test_data.cardinality(), reshuffle_each_iteration = True
        ).map(
            lambda image, label: dataframe.fetch_image_label_pair(image, label, self.architecture, augmentation, crop, crop_size),
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        ).batch(
            test_batch_size
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir = self.log_path,
            histogram_freq = 1,
            write_graph = True,
            write_images = True,
            write_steps_per_second = True,
            update_freq = 'epoch',
        )
        
        self.network.evaluate(
            test_data,
            batch_size = test_batch_size,
            verbose = 1,
            steps = None,
            callbacks = [
                EvaluationCustomCallback(self),
                tensorboard_callback
                ],
            return_dict = True
        )
        
    def predict(self, test_data, dataframe, test_batch_size, augmentation, crop, crop_size):
        """
        Obtain model's predicted labels on test data using model.predict()
        
        args:
        test_data (tf.data.Dataset): the test dataset
        dataframe (DataFrame): An instance of DataFrame containing the representation of the training and test data
        test_batch_size (int): the batch size of the test data
        """
        
        examples = test_data.map( # don't need to shuffle the test data, also allows to easily deternmine the true labels and predictions
            lambda image, label: dataframe.fetch_image_label_pair(image, label, self.architecture, True, crop, crop_size),
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        ).batch(
            test_batch_size
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )
        
        predictions = self.network.predict(
            x = examples,
            verbose = 1
        )
        
        print(predictions)
        self.save_predictions(predictions, test_data)
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
        Saved as: [checkpoint_path]/checkpoint-[timestamp]/model.weights.h5.
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
        
        args:
        path (str): the path to save the json file to
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
    
    def save_predictions(self, predictions, dataset):
        """
        Save the true labels and predicted labels in a dictionary and write it to a json file.
        Saved as {"image_path" : ([true_labels], [predicted_labels]), ...}
        """
        predictions_dir = os.path.join(self.predictions_path, f'predictions-{self.architecture}-{self.test_time}/')
        os.makedirs(predictions_dir, exist_ok = True)
        predictions_path = os.path.join(predictions_dir, 'predictions.json')
        keys_path = os.path.join(predictions_dir, 'keys.json')
        
        print(f"\033[33mSaving true labels and predictions to [{self.predictions_path}]... \033[0m")
        
        json_dict = {}
        keys = self.representation
        
        for element, (image_path, label) in enumerate(dataset):
            json_dict[image_path.numpy().decode('utf-8')] = (label.numpy().tolist(), predictions[element].tolist())
        
        print(len(json_dict), dataset.cardinality())
        
        json_obj = json.dumps(json_dict, indent = 4)
        keys_obj = json.dumps(keys, indent = 4)
        
        with open(predictions_path, 'w') as writer:
            writer.write(json_obj)
        
        with open(keys_path, 'w') as writer:
            writer.write(keys_obj)
            
        print("\033[32mSaved true labels and predictions! \033[0m")
        
    #----------------------------------------------------------------- NETWORK LOADING ----------------------------------------------------------------
    def load(self, network):
        """
        load weights and biases of the network from a file and the training state of the network from a file.
        
        args:
        network (tf.keras.Model): the network to load the weights and biases into
        """
        
        print(f"\033[32mLoaded training state: global_epoch = {self.global_epoch}, global_train_batch = {self.global_train_batch} \033[0m")
        weight_path = os.path.join(self.saved_weights_path)
        
        print(f"\033[33mLoading weights and biases from [{weight_path}]... \033[0m")
        network.load_weights(weight_path)  
        print("\033[32mLoaded weights and biases! \033[0m")  

#================================================================= CUSTOM CALLBACKS ======================================================================
    
class TrainingCustomCallbacks(keras.callbacks.Callback): # Inherits from keras.callbacks.Callback
        def __init__(self, NNreference):
            """
            Construct a custom callback to allow for custom callbacks during training,
            such as saving the model and writing metrics to a tensorboard log file.
            
            args:
            NNreference (Network): the Network being trained
            """
            super(TrainingCustomCallbacks, self).__init__() # calls the constructor of the parent class 
            self.NNreference = NNreference
            
        def on_batch_end(self, batch, logs = None): 
            """
            Save the metrics to the tensorboard log file every batch.
            
            args:
            batch (int): the batch number (expected argument by keras.callbacks.Callback)
            logs (dict): the logs to save
            """
            with self.NNreference.tensorboard_writer.as_default():
                # metrics per batch
                self.NNreference.metrics['loss'].append(logs.get('loss'))
                self.NNreference.metrics['accuracy'].append(logs.get('accuracy'))
                self.NNreference.metrics['root_mean_squared_error'].append(logs.get('root_mean_squared_error'))
                self.NNreference.metrics['mean_absolute_error'].append(logs.get('mean_absolute_error'))
                self.NNreference.metrics['binary_accuracy'].append(logs.get('binary_accuracy'))
                
                tf.summary.scalar(name = 'Loss', data = logs.get('loss'), step = self.NNreference.global_train_batch)
                tf.summary.scalar(name = 'Accuracy', data = logs.get('accuracy'), step = self.NNreference.global_train_batch)
                tf.summary.scalar(name = 'Root Mean Squared Error', data = logs.get('root_mean_squared_error'), step = self.NNreference.global_train_batch)
                tf.summary.scalar(name = 'Mean Absolute Error', data = logs.get('mean_absolute_error'), step = self.NNreference.global_train_batch)
                tf.summary.scalar(name = 'Binary Accuracy', data = logs.get('binary_accuracy'), step = self.NNreference.global_train_batch)
                self.NNreference.tensorboard_writer.flush()
                
                # save the model and metrics every save_step batches
                if self.NNreference.global_train_batch % self.NNreference.save_step == 0 and self.NNreference.global_train_batch != 0 and self.NNreference.global_train_batch % self.NNreference.steps_per_epoch != 0: # dont save when the save step is at the end or start of the epoch

                    self.NNreference.save(metrics = ['loss', 'accuracy', 'root_mean_squared_error', 'mean_absolute_error', 'binary_accuracy'])
                    
                self.NNreference.global_train_batch += 1
                
        def on_epoch_end(self, epoch, logs = None): 
            """
            Save the metrics to the tensorboard log file every epoch.
            
            args:
            epoch (int): the epoch number
            logs (dict): the logs to save
            """
            with self.NNreference.tensorboard_writer.as_default():
                # metrics per epoch
                self.NNreference.metrics['val_loss'].append(logs.get('val_loss'))
                self.NNreference.metrics['val_accuracy'].append(logs.get('val_accuracy'))
                
                tf.summary.scalar(name = 'Validation loss', data = logs.get('val_loss'), step = epoch)
                tf.summary.scalar(name = 'Validation accuracy', data = logs.get('val_accuracy'), step = epoch)
                self.NNreference.tensorboard_writer.flush()
                
                # save the model and metrics at the end of the epoch
                self.NNreference.save(metrics = ['val_loss', 'val_accuracy'])
                self.NNreference.global_epoch += 1

class EvaluationCustomCallback(tf.keras.callbacks.Callback): 
    def __init__(self, NNreference):
        """
        Construct a custom callback for use in model.evaluate()

        args:
        NNreference (Network): the Network being evaluated
        """       
        super(EvaluationCustomCallback, self).__init__()
        self.NNreference = NNreference
        
    def on_batch_end(self, batch, logs = None):
        """
        Save the metrics to the tensorboard log file every batch.
        
        args:
        batch (int): the batch number (expected argument by keras.callbacks.Callback)
        logs (dict): the logs to save
        """
        with self.NNreference.tensorboard_writer.as_default():
            # metrics per batch
            self.NNreference.metrics['loss'].append(logs.get('loss'))
            self.NNreference.metrics['accuracy'].append(logs.get('accuracy'))
            self.NNreference.metrics['root_mean_squared_error'].append(logs.get('root_mean_squared_error'))
            self.NNreference.metrics['mean_absolute_error'].append(logs.get('mean_absolute_error'))
            self.NNreference.metrics['binary_accuracy'].append(logs.get('binary_accuracy'))
            
            tf.summary.scalar(name = 'Loss', data = logs.get('loss'), step = self.NNreference.global_evaluate_batch)
            tf.summary.scalar(name = 'Accuracy', data = logs.get('accuracy'), step = self.NNreference.global_evaluate_batch)
            tf.summary.scalar(name = 'Root Mean Squared Error', data = logs.get('root_mean_squared_error'), step = self.NNreference.global_evaluate_batch)
            tf.summary.scalar(name = 'Mean Absolute Error', data = logs.get('mean_absolute_error'), step = self.NNreference.global_evaluate_batch)
            tf.summary.scalar(name = 'Binary Accuracy', data = logs.get('binary_accuracy'), step = self.NNreference.global_evaluate_batch)
            self.NNreference.tensorboard_writer.flush()

            if self.NNreference.global_evaluate_batch % self.NNreference.save_step == 0 and self.NNreference.global_evaluate_batch != 0 and self.NNreference.global_evaluate_batch % self.NNreference.steps_per_epoch != 0: # dont save when the save step is at the end or start of the epoch

                self.NNreference.save_evaluation_metrics() # saves evaluation metrics to csv
                    
            self.NNreference.global_evaluation_batch += 1
    
    def on_epoch_end(self, epoch, logs = None):
        """
        Save the metrics to the tensorboard log file every epoch.
        
        args:
        epoch (int): the epoch number
        logs (dict): the logs to save
        """
        with self.NNreference.tensorboard_writer.as_default():
            # metrics per epoch
            self.NNreference.metrics['val_loss'].append(logs.get('val_loss'))
            self.NNreference.metrics['val_accuracy'].append(logs.get('val_accuracy'))
            
            tf.summary.scalar(name = 'Validation loss', data = logs.get('val_loss'), step = epoch)
            tf.summary.scalar(name = 'Validation accuracy', data = logs.get('val_accuracy'), step = epoch)
            self.NNreference.tensorboard_writer.flush()
            
            # save the model and metrics at the end of the epoch
            self.NNreference.save_evaluation_metrics()
            self.NNreference.global_evaluation_epoch += 1
    