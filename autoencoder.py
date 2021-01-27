import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from IPython.display import SVG

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from IPython.display import Image
import datamerge
import pandas as pd

#%matplotlib inline

# FUNCTION: OUR MODEL

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> Batch Normalization -> ReLU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    ### END CODE HERE ###

    return model


def main():

    train_dataset_raw = datamerge.importDataset()
    train_dataset = pd.DataFrame({'label': train_dataset_raw.label.to_numpy()})
    data = train_dataset_raw.data.to_frame().applymap(lambda x: x[:, :12])#.to_numpy().reshape((len(train_dataset), 100, 12))
    sample = data.iloc[0].data
    data['label'] = train_dataset_raw.label.to_numpy()
    sample2 = data.iloc[0].label

    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()