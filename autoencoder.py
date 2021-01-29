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


def AutoencoderModel(input_shape):
    # Encoder
    X_input = Input(input_shape)
    X = ZeroPadding2D((4, 4))(X_input)
    X = Conv2D(16, (40, 3), strides=(4, 1), name='conv0')(X)
    X = Activation('relu')(X)
    X = Conv2D(32, (4, 4), strides=(1, 1), name='conv1')(X)
    X = Activation('relu')(X)
    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv2')(X)
    X = Flatten(name='flatten')(X)
    X = Activation('relu')(X)
    # Linear layers
    X = Dense(256, activation='relu', name='linear0')(X)
    X = Dense(16, activation=None, name='feature_out')(X)


    X = ReLU()(X)

    X = Dense(256, activation=None, name='linear1')(X)
    X = Dense(13*13*64, activation=None, name='linearReshape')(X)
    X = Reshape(target_shape=(13, 13, 64))(X)
    X = Conv2DTranspose(32, (3, 3), strides=(1, 1), name='convT0')(X)
    X = Activation('relu')(X)
    X = Conv2DTranspose(16, (4, 4), strides=(1, 1), name='convT1')(X)
    X = Activation('relu')(X)
    X = Conv2DTranspose(1, (40, 3), strides=(4, 1), name='convT2')(X)
    X = Cropping2D(cropping=(4, 4))(X)

    model = Model(inputs=X_input, outputs=X, name='AutoencoderModel')
    return model
def main():

    train_dataset_raw = datamerge.importDataset()
    # Extract MFCC
    train_dataset = pd.DataFrame({'label': train_dataset_raw.label.to_numpy()})
    data = train_dataset_raw.data.to_frame().applymap(lambda x: list(x[:, :12].flatten()))
    data['label'] = train_dataset_raw.label.to_numpy()

    # Create label dictionaries (35 different known words)
    un_labels = np.unique(data.label.to_numpy())    # All labels
    com_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']    # Command labels
    extra_labels = ['unknown word', 'silence']  # Special labels
    label_mask = np.isin(un_labels, com_labels)
    un_labels = un_labels[~label_mask]  # Non-command labels

    label_dict = dict(zip(com_labels, range(len(com_labels))))  # Commands oriented classification
    aut_label_dic = label_dict.copy()
    aut_label_dic.update(dict(zip(un_labels, range(len(un_labels)+len(com_labels)))))
    label_dict.update(dict(zip(extra_labels, range(len(extra_labels)+len(com_labels)))))    # All-words classifications

    title = train_dataset_raw.iloc[40]

    samples = np.array(data.data.tolist()).reshape((-1,100,12,1))
    sample = samples[0]
    in_shape = (100, 12, 1)
    #samples = np.empty((0, 100, 12, 1))
    #for i in range(len(data)):
    #        temp = np.expand_dims(data.iloc[i].data, 0)
    #       if (temp.shape==(1, 100, 12, 1)):
    #            samples = np.append(samples, temp, 0)

    autoenc = AutoencoderModel((in_shape))
    print(autoenc.summary())
    autoenc.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    autoenc.fit(x=samples, y=samples, epochs=4, batch_size=16)


    feat_out = autoenc.get_layer(name='feature_out').output
    in_x = autoenc.input
    encoder = Model(in_x, feat_out)
    print(encoder.summary())

    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()