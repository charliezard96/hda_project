import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Reshape, Conv2DTranspose, ReLU, Cropping2D

from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from IPython.display import SVG

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from IPython.display import Image
import datamerge
import pandas as pd

def Classifier(input_shape):

    X_input = Input(input_shape)
    # Linear layers
    X = Dense(1024, activation='relu', name='linear0')(X)
    X = Dense(2048, activation='relu', name='linear1')(X)
    X = Dense(11, activation=None, name='out')(X)

    model = Model(inputs=X_input, outputs=X, name='Classifier')
    return model