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



def AutoencoderModel(input_shape):
    # Encoder
    X_input = Input(input_shape)
    #X = ZeroPadding2D((4, 4))(X_input)
    X = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', name='conv0')(X_input)

    X = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', name='conv1')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv2')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv3')(X)
    X = Flatten(name='flatten')(X)

    # Linear layers
    X = Dense(1024, activation='relu', name='linear0')(X)
    X = Dense(128, activation=None, name='feature_out')(X)


    # Decoder
    X = Activation('relu')(X)
    X = Dense(1024, activation='relu', name='linear1')(X)
    X = Dense(92*4*64, activation=None, name='linearReshape')(X)
    X = Reshape(target_shape=(92, 4, 64))(X)
    X = Conv2DTranspose(64, (3, 3), strides=(1, 1), activation='relu', name='convT0')(X)
    X = Conv2DTranspose(32, (3, 3), strides=(1, 1), activation='relu', name='convT1')(X)
    X = Conv2DTranspose(16, (3, 3), strides=(1, 1), activation='relu', name='convT2')(X)
    X = Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='sigmoid', name='convT_out')(X)
    #X = Cropping2D(cropping=(4, 4))(X)
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

    samples = np.array(data.data.tolist()).reshape((-1, 100, 12, 1))
    sample = samples[0]
    in_shape = (100, 12, 1)
    #samples = np.empty((0, 100, 12, 1))
    #for i in range(len(data)):
    #        temp = np.expand_dims(data.iloc[i].data, 0)
    #       if (temp.shape==(1, 100, 12, 1)):
    #            samples = np.append(samples, temp, 0)

    autoenc = AutoencoderModel((in_shape))
    #autoenc = tf.keras.models.load_model('autoenc_first_train_gpu.h5')
    print(autoenc.summary())
    autoenc.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    #autoenc.fit(x=samples, y=samples, epochs=50, batch_size=256)
    #autoenc.save('autoenc_first_train_gpu.h5')

    feat_out = autoenc.get_layer(name='feature_out').output
    in_x = autoenc.input
    encoder = Model(in_x, feat_out)
    print(encoder.summary())

    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()