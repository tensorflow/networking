"""tensorflow_networking"""

import os
import tensorflow as tf

dirname = os.path.dirname(__file__)
tf.load_library(os.path.join(dirname, 'libtensorflow_networking.so'))
