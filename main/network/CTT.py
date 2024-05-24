import keras
from keras import layers
from keras import ops
import numpy as np
import tensorflow as tf

# credit for implementation: https://keras.io/examples/vision/cct/
class ConvolutionalTokeniser(layers.Layer):
    def __init__(self, kernel_size, stride, padding, pooling_kernel_size, pooling_stride, num_conv_layers, num_output_channels, positional_embedding):
        
        super().__init__()
        self.positional_embedding = positional_embedding
        self.conv_model = keras.Sequential()
        
        for i in range(num_conv_layers):
            
            self.conv_model.add(layers.Conv2D(num_output_channels[i], kernel_size, stride, padding = "valid", use_bias = False, activation = "relu", kernel_initializer = "he_normal"))
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(layers.MaxPooling2D(pooling_kernel_size, pooling_stride, "same"))

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through the convolutional tokeniser the spatial dimensions are flattened to form sequences.
        reshaped = keras.ops.reshape(
            outputs,
            (
                -1,
                keras.ops.shape(outputs)[1] * keras.ops.shape(outputs)[2],
                keras.ops.shape(outputs)[-1],
            ),
        )
        return reshaped
    
class PositionEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, initializer = "glorot_uniform"):
        super().__init__()
        
        if sequence_length is None:
            raise ValueError(f"\033[31m`sequence_length` must be an Integer! Received: `{type(sequence_length)}`.\033[0m")
        
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "initializer": keras.initializers.serialize(self.initializer)})
        return config

    def build(self, input_shape):
        
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(name = "embeddings", shape = [self.sequence_length, feature_size], initializer = self.initializer, trainable = True)

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        shape = keras.ops.shape(inputs)
        feature_length = shape[-1]
        sequence_length = shape[-2]
        # trim to match the length of the input sequence, as it might be less than the sequence_length of the layer.
        position_embeddings = keras.ops.convert_to_tensor(self.position_embeddings)
        position_embeddings = keras.ops.slice(position_embeddings, (start_index, 0), (sequence_length, feature_length))
        return keras.ops.broadcast_to(position_embeddings, shape)

    def compute_output_shape(self, input_shape):
        return input_shape
class SequencePooling(layers.Layer): # Sequence Pooling is used in Compact Covolutional Vision Transformers.
    def __init__(self):
        super().__init__()
        self.attention = layers.Dense(1)

    def call(self, x):
        attention_weights = keras.ops.softmax(self.attention(x), axis = 1)
        attention_weights = keras.ops.transpose(attention_weights, axes = (0, 2, 1))
        weighted_representation = keras.ops.matmul(attention_weights, x)
        return keras.ops.squeeze(weighted_representation, -2)
    
class StochasticDepth(layers.Layer): # Referred from: github.com:rwightman/pytorch-image-models.
    def __init__(self, drop_prop):
        super().__init__()
        self.drop_prob = drop_prop
        self.seed_generator = keras.random.SeedGenerator(0)

    def call(self, x, training = None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(shape, 0, 1, seed=self.seed_generator)
            random_tensor = keras.ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        
        return x