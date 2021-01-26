import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from IPython.display import SVG
from kt_utils import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from IPython.display import Image

%matplotlib inline
np.random.seed(1)