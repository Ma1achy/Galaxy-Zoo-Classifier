import keras
from keras import layers
from keras import ops
import numpy as np
import tensorflow as tf
from einops import rearrange
from tensorflow.keras import Input, Model

# implementation is HEAVILY based on the following :
# https://github.com/leoxiaobin/CvT/blob/main/lib/models/cls_cvt.py 
# ^ used in the following paper: https://arxiv.org/abs/2103.15808
# original implementation is in PyTorch, this is a TensorFlow implementation
# unfinished, only using MLP layer as of now
class MLP(layers.Layer):
    def __init__(self, hidden_units: list, dropout_rate: float):
        """
        Multi-layer perceptron layer, a feedforward neural network that consists of multiple layers of neurons.
        The input tensor x is passed through a series of dense layers with a GELU activation function and dropout layer.
        
        args:
        hidden_units (list): list of hidden units
        dropout_rate (float): dropout rate
        """
        super().__init__()
        self.layers = []
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        for units in hidden_units:
            self.layers.append(layers.Dense(units, activation = keras.activations.gelu, kernel_initializer = keras.initializers.RandomNormal(mean = 0.0, stddev = 0.05), bias_initializer = keras.initializers.Zeros()))
            self.layers.append(layers.Dropout(self.dropout_rate))
            
    def call(self, x: tf.Tensor): 
        """
        Forward pass of the multi-layer perceptron layer.
        
        args:
        x (tf.Tensor): input tensor
        
        returns:
        x (tf.Tensor): output tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
class LayerNorm(layers.Layer):
    def __init__(self):
        """
        Normalization layer that normalizes the input tensor x by subtracting the mean and dividing by the standard deviation.
        
        args:
        epsilon (float): small value to avoid division by zero
        """
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon = 1e-6, trainable = True)
           
    def call(self, x: tf.Tensor):
        """
        Forward pass of the normalization layer.
        
        args:
        x (tf.Tensor): input tensor 
        
        returns:
        x (tf.Tensor): output tensor
        """
        x = self.norm(x)
        return x

class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob: float):
        """
        Stochastic depth layer,
        A regularization technique that helps to prevent overfitting by randomly dropping a fraction of the input tensor during training.
        
        args:
        drop_prob (float): dropout probability
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.seed_generator = keras.random.SeedGenerator(0)
       
    def call(self, x: tf.Tensor, training: bool = False):
        """
        Forward pass of the stochastic depth layer.
        
        args:
        x (tf.Tensor): input tensor
        training (bool): whether the model is in training mode
        
        returns:
        x (tf.Tensor): output tensor
        """
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (keras.ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + keras.random.uniform(shape, 0, 1, seed=self.seed_generator)
            random_tensor = keras.ops.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        
        return x
    
class Attention(layers.Layer):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dropout_rate: float = 0, qkv_bias: bool = False, method: str = 'dw_bn', kernel_size: int = 3, stride_q: int = 1, stride_kv: int = 2, with_cls_token: bool = False, **kwargs):
        """
        Multi-head self-attention layer.
        Performs self-attention on the input tensor x with optional cls token by projecting the input tensor x to query, key and value tensors
        via a squeezed convolutional projection and then computes the attention score and output tensor.
        
        squeezed convolutional projection refers to the projection of the input tensor x to query, key and value tensors via a depthwise separable convolutional layer
        followed by a batch normalization layer. During the projection, the key and value tensors dimensions are reduced while the query tensor dimensions are kept the same,
        this results in a reduction of the number of parameters and computation required for the self-attention layer.
        
        args:
        dim_in (int): input dimension
        dim_out (int): output dimension
        num_heads (int): number of heads
        dropout_rate (float): dropout rate
        qkv_bias (bool): whether to include bias in qkv projection
        method (str): projection method to use
        with_cls_token (bool): whether to include cls token
        """
        
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self.__build_projection(
           dim_in, dim_out, kernel_size =  kernel_size, padding = 'same', stride = stride_q , method = ('linear' if method == 'avg' else method)
        )
        self.conv_proj_k = self.__build_projection(
            dim_in, dim_out, kernel_size = kernel_size, padding = 'valid', stride = stride_kv, method = method
        )
        self.conv_proj_v = self.__build_projection(
            dim_in, dim_out, kernel_size = kernel_size, padding = 'valid', stride = stride_kv, method = method
        )

        self.proj_q = layers.Dense(dim_out, use_bias = qkv_bias)
        self.proj_k = layers.Dense(dim_out, use_bias = qkv_bias)
        self.proj_v = layers.Dense(dim_out, use_bias = qkv_bias)

        self.attn_drop = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim_out)
        self.proj_drop = layers.Dropout(dropout_rate)

    def __build_projection(self, dim_in: int, dim_out: int, kernel_size: int, padding: str, stride: int, method: str):
        """
        Projection layer for query, key and value tensors.
        
        args:
        dim_in (int): input dimension
        dim_out (int): output dimension
        kernel_size (int): kernel size
        padding (str): padding type
        stride (int): the stride length
        method (str): projection method
        
        returns:
        proj (tf.keras.Sequential): projection layer
        """
        if method == "dw_bn":
            proj = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        dim_in,
                        kernel_size=kernel_size,
                        padding=padding,
                        strides=stride,
                        use_bias=False,
                        groups=dim_in,
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Reshape((-1, dim_in)),
                ]
            )
        elif method == 'average':
            proj = tf.keras.Sequential(
                [
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Reshape((-1, dim_in)),
                ]
            )
        elif method == 'linear':
            proj = None
        else:
            raise ValueError(f"Unknown method ({method})")
        
        return proj 
    
    def __convolve(self, x: tf.Tensor, h: int, w: int):
        """
        Perform convolutional projection on the input tensor x.
        
        args:
        x (tf.Tensor): input tensor
        h (int): height
        w (int): width
        """
        x = rearrange(x, 'b (h w) c -> b h w c', h = h, w = w) 
        if self.with_cls_token:
            cls_token, x = tf.split(x, [1, h * w], axis = 1)
        
        if self.conv_proj_q is not None:
            x = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b h w c -> b (h w) c')
        
        if self.conv_proj_k is not None:
            x = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b h w c -> b (h w) c')
            
        if self.conv_proj_v is not None:
            x = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b h w c -> b (h w) c')
        
        if self.with_cls_token:
            q = tf.concat([cls_token, q], axis = 1)
            k = tf.concat([cls_token, k], axis = 1)
            v = tf.concat([cls_token, v], axis = 1)
          
        return q, k, v
        
    def call(self, x: tf.Tensor, h: int, w: int, training: bool = False):
        """
        Forward pass of the attention layer.
        
        args:
        x (tf.Tensor): input tensor
        h (int): height
        w (int): width
        training (bool): whether the model is in training mode
        
        returns:
        x (tf.Tensor): output tensor
        """
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.__convolve(x, h, w)
        
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h = self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h = self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h = self.num_heads)
        
        attention_score = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention_score = tf.nn.softmax(attention_score, axis = -1)
        attention_score = self.attn_drop(attention_score, training = training)
        
        x = tf.einsum('b h i j, b h j d -> b h i d', attention_score, v)
        x = rearrange(x, 'b h t d -> b t (h d)', h = self.num_heads)
        
        x = self.proj(x)
        x = self.proj_drop(x, training = training)
        
        return x
   
class ConvolutionalEmbedding(layers.Layer):
    def __init__(self, padding: str, patch_size: int, embed_dim: int, stride: int, norm_layer: bool, **kwargs):
        """
        Perform convolutional projection on the input tensor x to create a tokenized representation of the input tensor.
        The input tensor x is projected to a lower-dimensional tensor via a convolutional layer followed by a layer normalization layer,
        this results in a reduction of the token sequence length while increasing the token dimensions, giving the tokens a richer representation of
        increasingly complex features over larger spatial regions.
    
        args:
        padding (str): padding type
        patch_size (int): patch size
        embed_dim (int): embedding dimension, determines the number of convolutional filters
        stride (int): stride
        norm_layer (bool): whether to include normalization layer
        """
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.patch_size = (patch_size, patch_size)
        self.projection = layers.Conv2D(embed_dim, kernel_size = self.patch_size, strides = stride, padding = padding, use_bias = False)
        
        if norm_layer:
            self.norm = layers.LayerNormalization(epsilon = 1e-6, trainable = True) 
        else:
            self.norm = None
     
    def call(self, x: tf.Tensor):
        """
        Forward pass of the convolutional embedding layer.
        
        args:
        x (tf.Tensor): input tensor

        returns:
        x (tf.Tensor): output tensor
        """
        x = self.projection(x)
        B, W, H, C = x.shape
             
        x = rearrange(x, 'b h w c -> b (h w) c')
     
        if self.norm:
            x = self.norm(x)
        
        x = rearrange(x, 'b (h w) c -> b h w c', h = H, w = W)
        
        return x
    
class Block(layers.Layer):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, drop_path_rate: float, drop_prob: float, dropout_rate: float, method: str, with_cls_token: bool, **kwargs):
        """
        Block layer, a single transformer block that consists of a multi-head self-attention layer and a multi-layer perceptron layer.
        The input tensor x is passed through a layer normalization layer, multi-head self-attention layer, and a multi-layer perceptron layer.
        The output tensor is then passed through a dropout layer and a skip connection to the input tensor.
        
        args:
        dim_in (int): input dimension
        dim_out (int): output dimension
        num_heads (int): number of heads
        mlp_ratio (float): mlp ratio
        qkv_bias (bool): whether to include bias in qkv projection
        drop_path_rate (float): drop path rate
        drop_prob (float): dropout probability
        dropout_rate (float): dropout rate
        method (str): projection method
        with_cls_token (bool): whether to include a cls token
        """
        super().__init__()
        
        self.with_cls_token = with_cls_token
        self.norm1 = LayerNorm(dim_in)
        self.attention = Attention(
            dim_in, dim_out, num_heads, dropout_rate, qkv_bias, method, self.with_cls_token
        )
        
        self.drop_path = StochasticDepth(drop_prob) if drop_path_rate > 0 else tf.identity
            
        self.norm2 = LayerNorm(dim_out)
        
        dim_mlp_hidden = int(dim_out * mlp_ratio)
        
        self.mlp = MLP([dim_mlp_hidden, dim_out], dropout_rate)
        
        def call(self, x: tf.Tensor, h: int, w: int, training: bool = False):
            """
            Forward pass of the block layer.
            
            args:
            x (tf.Tensor): input tensor
            h (int): height
            w (int): width
            training (bool): whether the model is in training mode
            
            returns:
            x (tf.Tensor): output tensor
            """
            res = x
            x = self.norm1(x)
            x = self.attention(x, h, w, training = training)
            x = self.drop_path(x, training = training)
            x = res + x
            return x

class VisionTransformer(layers.Layer):
    def __init__(self, patch_size: int, patch_stride: int, patch_padding: str, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, drop_path_rate: float, dropout_rate: float, method: str, with_cls_token: bool, norm_layer: bool, init: str, **kwargs):
        """
        Vision transformer model, a convolutional neural network that consists of multiple transformer blocks.
        The input tensor x is passed through a patch embedding layer, positional dropout layer, and multiple transformer blocks.
        The output tensor is then passed through a normalization layer and a classification layer.
        
        args:
        patch_size (int): patch size
        patch_stride (int): patch stride
        patch_padding (str): patch padding
        embed_dim (int): embedding dimension
        depth (int): depth
        num_heads (int): number of heads
        mlp_ratio (float): mlp ratio
        qkv_bias (bool): whether to include bias in qkv projection
        drop_path_rate (float): drop path rate
        dropout_rate (float): dropout rate
        method (str): projection method
        with_cls_token (bool): whether to include a cls token
        norm_layer (bool): whether to include normalization layer
        init (str): initialization method
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.rearrage = None
        
        self.patch_embed = ConvolutionalEmbedding(
            patch_size=patch_size,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        
        if with_cls_token:
            self.cls_token = self.add_weight(
                "cls_token",
                shape=(1, 1, embed_dim),
                initializer=keras.initializers.Zeros(),
                trainable=True,
            )
        else:
            self.cls_token = None
        
        self.pos_drop = layers.Dropout(dropout_rate)
        dpr = [x for x in np.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule
        
        self.blocks = []
        
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=dpr[i],
                    drop_prob=dropout_rate,
                    dropout_rate=dropout_rate,
                    method=method,
                    with_cls_token=with_cls_token,
                    **kwargs
                )
            )
        
        if self.cls_token is not None:
            initializer = tf.initializers.TruncatedNormal(stddev=0.02)
            self.cls_token = self.add_weight(
            "cls_token",
            shape=(1, 1, embed_dim),
            initializer=initializer,
            trainable=True,
        )
            
        if init == "xavier":
            self.apply(self.__init_weights_xavier)
        else:
            self.apply(self.__init_weights_trunc_normal)
        
    def __init_weights_trunc_normal(self, layer: tf.keras.layers.Layer):
        """
        Truncated normal initialization, a weight initialization technique that initializes the weights of the neural network to small random values.
        
        args:
        layer (tf.keras.layers.Layer): layer to initialize
        """
        if isinstance(layer, layers.Dense):
            layer.kernel.initializer = tf.initializers.TruncatedNormal(stddev=0.02)
            
            if layer.bias is not None:
                layer.bias.initializer = tf.initializers.Zeros()
                
        elif isinstance(layer, (layers.LayerNormalization, layers.BatchNormalization)):
            layer.gamma.initializer = tf.initializers.Ones()
            layer.beta.initializer = tf.initializers.Zeros()
        
    def __init_weights_xavier(self, layer: tf.keras.layers.Layer):
        """
        Xavier initialization, a weight initialization technique that initializes the weights of the neural network to small random values.
        
        args:
        layer (tf.keras.layers.Layer): layer to initialize
        """
        if isinstance(layer, layers.Dense):
            layer.kernel.initializer = tf.initializers.GlorotNormal()
            
            if layer.bias is not None:
                layer.bias.initializer = tf.initializers.Zeros()
                
        elif isinstance(layer, (layers.LayerNormalization, layers.BatchNormalization)):
            layer.gamma.initializer = tf.initializers.Ones()
            layer.beta.initializer = tf.initializers.Zeros()
    
    def call(self, x: tf.Tensor, training: bool = False):
        """
        Forward pass of the vision transformer model.
        
        args:
        x (tf.Tensor): input tensor
        training (bool): whether the model is in training mode
        
        returns:
        x (tf.Tensor): output tensor
        cls_tokens (tf.Tensor): cls token tensor
        """
        x = self.patch_embed(x)
        B, H, W, C = x.shape
        
        x = rearrange(x, 'b h w c -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None: # cls_tokens implementation from Phil Wang
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = tf.concat([cls_tokens, x], axis = 1)
        
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x, H, W, training = training)
        
        if self.cls_token is not None:
            cls_tokens, x = tf.split(x, [1, H * W], axis = 1)
        
        x = rearrange(x, 'b (h w) c -> b h w c', h = H, w = W)
        
        return x, cls_tokens
    
class ConvolutionalVisionTransformer(layers.Layer):
    def __init__(self, in_chans: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = spec['NUM_STAGES']
        
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['EMBED_DIM'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'dropout_rate': spec['DROPOUT_RATE'][i],
                'method': spec['METHOD'][i],
                'with_cls_token': spec['WITH_CLS_TOKEN'][i],
                'norm_layer': spec['NORM_LAYER'][i],
                'init': spec['INIT'][i]
            }