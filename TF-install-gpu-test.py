import tensorflow as tf
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# NOTE:
# This script is for testing if TensorFlow is installed with GPU support.
# Native GPU support on Windows was removed in versions > 2.10 :(
# Using TensorFlow 2.10 with CUDA 11.2 and cuDNN 8.1 with Python 3.10

