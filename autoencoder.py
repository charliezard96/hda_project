import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Reshape, Conv2DTranspose, ReLU, Cropping2D, Add

from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from IPython.display import SVG

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from IPython.display import Image
import datamerge
import pandas as pd
from tensorflow.keras.initializers import glorot_uniform
def extractData_noisly(dataset_raw):
    data = dataset_raw.data.to_frame().applymap(lambda x: list(x[:, :12].flatten()))
    #data['label'] = dataset_raw.label.to_numpy()
    data['noisy'] = data.data.to_frame().applymap(add_noise)
    samples = np.array(data.data.tolist()).reshape((-1, 100, 12, 1))
    noisy_samples = np.array(data.noisy.tolist()).reshape((-1, 100, 12, 1))
    # Non ci servono qua
    '''
    # Create label dictionaries (35 different known words)
    com_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']  # Command labels
    label_dict = dict(zip(com_labels, range(len(com_labels))))  # Commands oriented classification

    
    # Non ci servono qua
    #true_labels = data['label'].to_frame().applymap(lambda x: label_dict.get(x, len(label_dict))).to_numpy() 
    '''
    return data, samples, noisy_samples

def add_noise(src, var = 0.1):
    src_shape = len(src)
    sigma = var ** 0.5
    noise = np.random.normal(0, sigma, src_shape)

    return src + noise

def AutoencoderModel(input_shape):
    # Encoder
    X_input = Input(input_shape)

    X = Conv2D(16, (3, 3), strides=(1, 1), activation='elu', name='conv0', padding='same')(X_input)

    X = Conv2D(32, (3, 3), strides=(1, 1), activation='elu', name='conv1', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), activation='elu', name='conv2', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), activation='elu', name='conv3', padding='same')(X)

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

def AutoencoderModel_withBatch(input_shape):
    # Encoder
    X_input = Input(input_shape)

    X = Conv2D(8, (3, 3), strides=(1, 1), name='conv0', padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(16, (3, 3), strides=(1, 1), name='conv1', padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv2', padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv3', padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Flatten(name='flatten')(X)

    # Linear layers
    X = Dense(1024, activation='relu', name='linear0')(X)
    X = Dense(512, activation='relu', name='linear1')(X)
    X = Dense(256, activation=None, name='feature_out')(X)

    # Decoder
    X = Activation('relu')(X)
    X = Dense(512, activation='relu', name='invLinear1')(X)
    X = Dense(1024, activation='relu', name='inLinear0')(X)
    X = Dense(100 * 12 * 32, activation=None, name='linearReshape')(X)
    X = Reshape(target_shape=(100, 12, 32))(X)

    X = Conv2DTranspose(32, (3, 3), activation='relu', strides=(1, 1), name='convT0', padding='same')(X)

    X = Conv2DTranspose(16, (3, 3), activation='relu', strides=(1, 1), name='convT1', padding='same')(X)

    X = Conv2DTranspose(8, (3, 3), activation='relu', strides=(1, 1), name='convT2', padding='same')(X)

    X = Conv2DTranspose(1, (3, 3), activation=None, strides=(1, 1), name='convT_out', padding='same')(X)

    model = Model(inputs=X_input, outputs=X, name='AutoencoderModel')

    return model

def AutoencoderModel_withSC(input_shape):
    # Encoder
    X_input = Input(input_shape)

    X = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', name='conv0', padding='same')(X_input)

    X = Conv2D(32, (3, 3), strides=(1, 1), activation='relu', name='conv1', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv2', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', name='conv3', padding='same')(X)

    ## SHORTCUT
    X_shortcut = Conv2D(64, (1, 1), strides=(4, 1), padding='same', name='shortcut')(X_input)
    X_shortcut = BatchNormalization(axis=3, name='bn1')(X_shortcut)
    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

    X = Flatten(name='flatten')(X)

    # Linear layers
    X = Dense(1024, activation='relu', name='linear0')(X)
    X = Dense(512, activation='relu', name='linear1')(X)
    X = Dense(256, activation=None, name='feature_out')(X)

    # Decoder
    X = Activation('relu')(X)
    X = Dense(512, activation='relu', name='invLinear1')(X)
    X = Dense(1024, activation='relu', name='inLinear0')(X)
    X = Dense(25 * 12 * 64, activation=None, name='linearReshape')(X)
    X = Reshape(target_shape=(25, 12, 64))(X)

    X = Activation('relu')(X)

    X = Conv2DTranspose(64, (3, 3), activation='relu', strides=(1, 1), name='convT0', padding='same')(X)

    X = Conv2DTranspose(32, (3, 3), activation='relu', strides=(2, 1), name='convT1', padding='same')(X)

    X = Conv2DTranspose(16, (3, 3), activation='relu', strides=(2, 1), name='convT2', padding='same')(X)

    X = Conv2DTranspose(1, (3, 3), activation=None, strides=(1, 1), name='convT_out', padding='same')(X)

    model = Model(inputs=X_input, outputs=X, name='AutoencoderModel')
    return model

def AutoencoderModel_withSCandBN(input_shape):
    # Encoder
    X_input = Input(input_shape)

    X = Conv2D(4, (3, 3), strides=(1, 1), name='conv0', padding='same', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(8, (3, 3), strides=(1, 1), name='conv1', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(16, (3, 3), strides=(1, 1), name='conv2', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', name='conv3', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPooling2D(pool_size=(2, 1), padding='same')(X)

    ## SHORTCUT
    X_shortcut = Conv2D(16, (1, 1), strides=(4, 1), padding='same', name='shortcut_conv', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X_shortcut = BatchNormalization(axis=3, name='shortcut_bn')(X_shortcut)
    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

    X = Flatten(name='flatten')(X)

    # Linear layers
    X = Dense(1024, activation='relu', name='linear0')(X)
    X = Dense(512, activation='relu', name='linear1')(X)
    X = Dense(256, activation=None, name='feature_out')(X)

    # Decoder
    X = Activation('relu')(X)
    X = Dense(512, activation='relu', name='invLinear1')(X)
    X = Dense(1024, activation='relu', name='inLinear0')(X)
    X = Dense(25 * 12 * 16, activation=None, name='linearReshape')(X)
    X = Reshape(target_shape=(25, 12, 16))(X)

    X = Activation('relu')(X)

    X = Conv2DTranspose(16, (3, 3), activation='relu', strides=(2, 1), name='convT0', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = Conv2DTranspose(8, (3, 3), activation='relu', strides=(2, 1), name='convT1', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = Conv2DTranspose(4, (3, 3), activation='relu', strides=(1, 1), name='convT2', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = Conv2DTranspose(1, (3, 3), activation=None, strides=(1, 1), name='convT_out', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='AutoencoderModel_new')
    return model

def main():

    train_dataset_raw = datamerge.importDataset('dTrain2')
    val_dataset_raw = pd.read_hdf('dVal.h5')
    """
    print('### Train data processing ###')
    data, samples, true_labels = extractData(train_dataset_raw)
    print('### Validation data processing ###')
    val_data, val_samples, val_labels = extractData(val_dataset_raw)
    """
    """ PARTE VECCHIA
    train_dataset_raw = datamerge.importDataset('dTrain2')
    val_dataset_raw = pd.read_hdf('dVal.h5')
    # Extract MFCC
    train_dataset = pd.DataFrame({'label': train_dataset_raw.label.to_numpy()})
    data = train_dataset_raw.data.to_frame().applymap(lambda x: list(x[:, :12].flatten()))
    data['label'] = train_dataset_raw.label
    data['noisy'] = data.data.to_frame().applymap(add_noise)
    samples = np.array(data.data.tolist()).reshape((-1, 100, 12, 1))
    noisy_samples = np.array(data.noisy.tolist()).reshape((-1, 100, 12, 1))
    sample = samples[0]
    """

    data, samples, noisy_samples = extractData_noisly(train_dataset_raw)
    val_data, val_samples, val_noisy = extractData_noisly(val_dataset_raw)
    in_shape = (100, 12, 1)
    # autoenc = tf.keras.models.load_model('autoencoders_models\\autoenc_gpu_500.h5')

    ### MODEL DEFINITION
    autoenc = AutoencoderModel((in_shape))
    stringa = "AutoencoderModel"
    print(autoenc.summary())
    #tf.keras.utils.plot_model(autoenc, to_file='graph\\'+stringa+'.png')

    ### MODEL FIT
    autoenc.compile(optimizer="adam", loss="mean_squared_error")
    history = autoenc.fit(x=noisy_samples, y=samples, epochs=2, batch_size=256)
    plt.plot(history.history['loss'])
    plt.show()

    ### MODEL SAVE
    #autoenc.save('autoencoders_models\\'+stringa+'_train.h5')

    #with open("history\\history"+stringa+".txt", "w") as output:
        #output.write(str(history.history['loss']))

    ### ENCODER SAVE
    #feat_out = autoenc.get_layer(name='feature_out').output
    #in_x = autoenc.input
    #encoder = Model(in_x, feat_out)
    #encoder.save('encoder_gpu_500.h5')
    #print(encoder.summary())

    print('end of the job')
    z = 1  # Linea di debugging


if __name__ == "__main__":
    main()