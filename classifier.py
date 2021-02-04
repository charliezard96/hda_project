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

def main():

    train_dataset_raw = datamerge.importDataset()
    # Extract MFCC
    train_dataset = pd.DataFrame({'label': train_dataset_raw.label.to_numpy()})
    data = train_dataset_raw.data.to_frame().applymap(lambda x: list(x[:, :12].flatten()))
    data['label'] = train_dataset_raw.label.to_numpy()
    samples = np.array(data.data.tolist()).reshape((-1, 100, 12, 1))

    # Create label dictionaries (35 different known words)
    un_labels = np.unique(data.label.to_numpy())  # All labels
    com_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']  # Command labels
    extra_labels = ['unknown word', 'silence']  # Special labels
    label_mask = np.isin(un_labels, com_labels)
    un_labels = un_labels[~label_mask]  # Non-command labels

    label_dict = dict(zip(com_labels, range(len(com_labels))))  # Commands oriented classification
    aut_label_dic = label_dict.copy()
    aut_label_dic.update(dict(zip(un_labels, range(len(un_labels) + len(com_labels)))))
    label_dict.update(dict(zip(extra_labels, range(len(extra_labels) + len(com_labels)))))

    z = 1  # Linea di debugging


if __name__ == "__main__":
    main()