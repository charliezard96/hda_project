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
def add_noise(src):
    src_shape = len(src)
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    noise = np.random.normal(0, sigma, src_shape)

    return src + noise

def AutoencoderModel(input_shape):
    # Encoder
    X_input = Input(input_shape)

    X = Conv2D(16, (3, 3), strides=(1, 1), activation='elu', name='conv0', padding='same')(X_input)
    #  X = MaxPooling2D(pool_size=(3, 3), padding='same')(X)

    X = Conv2D(32, (3, 3), strides=(1, 1), activation='elu', name='conv1', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), activation='elu', name='conv2', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), activation='elu', name='conv3', padding='same')(X)
    # X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)

    X = Flatten(name='flatten')(X)

    # Linear layers
    X = Dense(1024, activation='elu', name='linear0')(X)
    X = Dense(512, activation='elu', name='linear1')(X)
    X = Dense(256, activation=None, name='feature_out')(X)

    # Decoder
    X = Activation('elu')(X)
    X = Dense(512, activation='elu', name='invLinear1')(X)
    X = Dense(1024, activation='elu', name='inLinear0')(X)
    X = Dense(25 * 12 * 64, activation=None, name='linearReshape')(X)
    X = Reshape(target_shape=(25, 12, 64))(X)

    X = Conv2DTranspose(64, (3, 3), activation='elu', strides=(1, 1), name='convT0', padding='same')(X)

    X = Conv2DTranspose(32, (3, 3), activation='elu', strides=(2, 1), name='convT1', padding='same')(X)

    X = Conv2DTranspose(16, (3, 3), activation='elu', strides=(2, 1), name='convT2', padding='same')(X)

    X = Conv2DTranspose(1, (3, 3), activation=None, strides=(1, 1), name='convT_out', padding='same')(X)

    model = Model(inputs=X_input, outputs=X, name='AutoencoderModel')
    return model
def main():

    train_dataset_raw = datamerge.importDataset()
    #train_dataset_raw = pd.read_hdf('dVal.h5')
    # Extract MFCC
    train_dataset = pd.DataFrame({'label': train_dataset_raw.label.to_numpy()})
    data = train_dataset_raw.data.to_frame().applymap(lambda x: list(x[:, :12].flatten()))
    data['label'] = train_dataset_raw.label
    data['noisy'] = data.data.to_frame().applymap(add_noise)
    samples = np.array(data.data.tolist()).reshape((-1, 100, 12, 1))
    noisy_samples = np.array(data.noisy.tolist()).reshape((-1, 100, 12, 1))
    sample = samples[0]
    in_shape = (100, 12, 1)

    autoenc = AutoencoderModel((in_shape))
    #autoenc = tf.keras.models.load_model('autoencoders_models\\autoenc_gpu_500.h5')
    print(autoenc.summary())

    autoenc.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    history = autoenc.fit(x=noisy_samples, y=samples, epochs=150, batch_size=256)
    plt.plot(history.history['loss'])
    plt.show()

    autoenc.save('autoencoders_models\\autoenc_gpu_500.h5')

    feat_out = autoenc.get_layer(name='feature_out').output
    in_x = autoenc.input
    '''
    encoder = Model(in_x, feat_out)
    encoder.save('encoder_gpu_500.h5')
    print(encoder.summary())
    '''
    z = 1  # Linea di debugging

if __name__ == "__main__":
    main()