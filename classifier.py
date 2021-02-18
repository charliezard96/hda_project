import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Reshape, Conv2DTranspose, ReLU, Cropping2D
from sklearn import decomposition, model_selection, metrics, manifold

from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from IPython.display import SVG

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from IPython.display import Image
import datamerge
import pandas as pd

def extractData(dataset_raw):
    data = dataset_raw.data.to_frame().applymap(lambda x: list(x[:, :12].flatten()))
    data['label'] = dataset_raw.label.to_numpy()

    # Create label dictionaries (35 different known words)
    com_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']  # Command labels
    label_dict = dict(zip(com_labels, range(len(com_labels))))  # Commands oriented classification

    samples = np.array(data.data.tolist()).reshape((-1, 100, 12, 1))
    true_labels = data['label'].to_frame().applymap(lambda x: label_dict.get(x, len(label_dict))).to_numpy()
    return(data, samples, true_labels)

def Classifier(input_shape):

    X_input = Input(input_shape)
    # Linear layers
    X = Dense(1024, activation='relu', name='linear0')(X_input)
    X = Dense(2048, activation='relu', name='linear1')(X)
    X = Dense(11, activation='softmax', name='out')(X)

    model = Model(inputs=X_input, outputs=X, name='Classifier')
    return model

def main():

    train_dataset_raw = datamerge.importDataset('dTrain2')
    # train_dataset_raw = pd.read_hdf('dVal.h5')
    val_dataset_raw = pd.read_hdf('dVal.h5')

    print('### Train data processing ###')
    data, samples, true_labels = extractData(train_dataset_raw)
    print('### Validation data processing ###')
    val_data, val_samples, val_labels = extractData(val_dataset_raw)


    # model

    encoder = tf.keras.models.load_model('autoencoders_models\\encoder_gpu_150_noise.h5')
    print(encoder.summary())
    in_shape = (100, 12, 1)
    inp = tf.keras.Input(in_shape)
    features = encoder(inp)
    n_features = 256
    classification = Classifier(n_features)
    prediction = classification(features)

    predictor = tf.keras.Model(inputs=inp, outputs=prediction)
    predictor.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    history = predictor.fit(x=samples, y=true_labels, epochs=50, batch_size=256)
    #predictor.save('autoencoders_models\\classifier_50_noise.h5')
    predicted_labels = predictor.predict(samples)


    ### Confusion matrix
    # Predicted labels
    y_true = true_labels
    y_pred = np.argmax(predicted_labels, axis=1)
    # Evaluate confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Show the confusion matrix
    pd.DataFrame(cm)
    print(pd.DataFrame(cm).head(64))
    z = 1  # Linea di debugging


if __name__ == "__main__":
    main()