import keras
from keras import layers
from keras import ops
import numpy as np
import tensorflow as tf
# credit for implementation: https://keras.io/examples/vision/image_classification_with_vision_transformer/#build-the-vit-model

def mlp(x, hidden_units, dropout_rate):
    """
    Multilayer Perceptron (MLP)
    
    args:
    x (keras.layers.Layer): the input layer
    hidden_units (list): a list of integers representing the number of units in each hidden layer
    dropout_rate (float): the dropout rate
    
    returns:
    x (keras.layers.Layer): the output layer
    """
    for units in hidden_units:
        x = layers.Dense(units, activation = keras.activations.gelu, kernel_initializer = 'he_uniform', bias_initializer = keras.initializers.Zeros())(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def augmentation_layers(input_x, input_y):
    """
    Image augmentation layers.
    
    args:
    input_x (int): the shape of the input in the x dimension
    input_y (int): the shape of the input in the y dimension
    
    returns:
    data_augmentation (keras.Sequential); the data augmentation layers
    """
    data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(factor = (-1, 1), fill_mode = 'constant', fill_value = 0.0),
        keras.layers.RandomFlip(mode="horizontal_and_vertical"),
        layers.RandomZoom(height_factor = (-5/input_y, 5/input_y), width_factor = (-5/input_x, 5/input_x), fill_mode = 'constant', fill_value = 0.0),
        layers.RandomTranslation(height_factor = (-5/input_y, 5/input_y), width_factor = (-5/input_x, 5/input_x), fill_mode = 'constant', fill_value = 0.0)
    ],
    name="data_augmentation",
    )
    return data_augmentation

# ============================================================================================== Vision Transformer ==============================================================================================
class Patches(layers.Layer):
    def __init__(self,  input_x, num_patches):
        """
        Construct a keras layer that splits images into patches to be passed into a transformer model
        
        args:
        input_x (int): the side length of the image
        num_patches (int): the number of patches to split the image into
        """
        super().__init__()
        self.num_patches = int(np.sqrt(num_patches))
        self.patch_size = input_x // self.num_patches
        
    def call(self, images):
        """
        Create patches from the input image tensor
        
        args:
        images (keras.layers.Layer): the input image tensor
        
        returns:
        patches (keras.layers.Layer): the patches of the image
        """
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        size = input_shape[1]  
        channels = input_shape[3]
        num_patches = size // self.patch_size

        patches = keras.ops.image.extract_patches(images, size = self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches**2,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        """
        Layer configuration
        
        returns:
        config (dict): the configuration of the layer
        """
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        """
        Construct a Patch Encoding Layer.
        Linearly transforms a patch by projecting it into a vector of size projection_dim and adds a learnable position 
        embedding to the projected vector.
        
        args:
        num_patches (int): the total number of patches the image is divided into
        projection_dim (int): the dimensionality of the projected vector space.
        """
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units = projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim = num_patches, output_dim = projection_dim
        )

    def call(self, patch):
        """
        Encodes the patch by projecting it into a vector of size projection_dim and adding a learnable position embedding.
        
        args:
        patch (keras.layers.Layer): the patch to be encoded
        
        returns:
        encoded (keras.layers.Layer): the encoded patch
        """
        positions = ops.expand_dims(
            ops.arange(start=0, stop = self.num_patches, step = 1), axis = 0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        """
        Layer configuration
        
        returns:
        config (dict): the configuration of the layer
        """
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
#============================================================================================== Convolutional Vision Transformer ==============================================================================================
class ConvolutionalEmbedding(layers.Layer):
    def __init__(self, patch_size, embed_dim, stride, padding, norm_layer):
        super().__init__()
        self.patch_size = patch_size
        self.proj = layers.Conv2D(embed_dim, patch_size, strides=stride, padding=padding, use_bias = False, activation = None, kernel_initializer = 'he_normal')
        self.norm = layers.LayerNormalization(epsilon = 1e-6, trainable = True) if norm_layer else None

    def call(self, x):
        x = self.proj(x)
        x = layers.Reshape((-1, x.shape[-1]))(x)
        if self.norm:
            x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'patch_size': self.patch_size, 'proj': self.proj, 'norm': self.norm})
        return config

class SeparableConvMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout, kernel_size, stride, padding):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.depthwise_conv = layers.SeparableConv1D(filters = key_dim, kernel_size = kernel_size, strides = stride, padding = padding, activation = None, use_bias = False, depthwise_initializer = 'he_normal', pointwise_initializer = 'he_normal')
        self.multi_head_attention = layers.MultiHeadAttention(num_heads = num_heads, key_dim = key_dim, dropout = dropout)

    def call(self, inputs):
        # Apply depth-wise separable convolution before multi-head attention.
        x = self.depthwise_conv(inputs)
        return self.multi_head_attention(x, x)
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_heads': self.num_heads, 'key_dim': self.key_dim, 'dropout': self.dropout, 'depthwise_conv':  self.depthwise_conv, 'multi_head_attention': self.multi_head_attention})
        return config
           
class AddCLSToken(layers.Layer):
    def __init__(self, cls_token, **kwargs):
        super().__init__(**kwargs)
        self.cls_token = tf.Variable(cls_token, trainable=True)

    def call(self, inputs):
        cls_tokens = tf.repeat(self.cls_token[tf.newaxis, :], tf.shape(inputs)[0], axis=0)
        return tf.concat([cls_tokens, inputs], axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({'cls_token': self.cls_token})
        return config
    