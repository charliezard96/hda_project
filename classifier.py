import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    Reshape, Conv2DTranspose, ReLU, Cropping2D
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
    return (data, samples, true_labels)


def Classifier(input_shape, l1, l2):
    X_input = Input(input_shape)
    # Linear layers
    X = Dense(l1, activation='relu', name='linear0')(X_input)
    X = Dense(l2, activation='relu', name='linear1')(X)
    X = Dropout(rate=0.2)(X)
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
    print('###Model initialization###')
    autoenc = tf.keras.models.load_model('autoencoders_models\\AUTOENCODER\\AutoencoderModel_withBatch150_train.h5')

    feat_out = autoenc.get_layer(name='feature_out').output
    in_x = autoenc.input
    encoder = Model(in_x, feat_out)
    # print(encoder.summary())
    in_shape = (100, 12, 1)

    print('###forwarding through encoder the dataset###')
    features = encoder.predict(samples)
    val_features = encoder.predict(val_samples)

    n_features = 256
    l1 = 1024
    l2 = 1024
    num_epochs = 300
    from_AE = "ABN"
    stringa = "DROP_classifier_"+from_AE+"_"+str(l1)+"_"+str(l2)+"_epochs"+str(num_epochs)
    predictor = Classifier(n_features, l1, l2)

    predictor.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = predictor.fit(x=features, y=true_labels, validation_data=(val_features, val_labels), epochs=num_epochs, batch_size=256)

    ### MODEL SAVE
    predictor.save('classifiers\\' + stringa + '.h5')
    ### HISTORY SAVE
    with open("history\\CLASSIFIER\\loss_history_"+stringa+".txt", "w") as output:
        output.write(str(history.history['loss']))
    with open("history\\CLASSIFIER\\val_loss_history_"+stringa+".txt", "w") as output:
        output.write(str(history.history['val_loss']))
    with open("history\\CLASSIFIER\\acc_history_" + stringa+ ".txt","w") as output:
        output.write(str(history.history['accuracy']))
    with open("history\\CLASSIFIER\\val_acc_history_" + stringa+ ".txt","w") as output:
        output.write(str(history.history['val_accuracy']))

    ### PREDICTION
    predicted_labels = predictor.predict(features)

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
