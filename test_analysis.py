import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.models import Model

import pandas as pd

import random

def extractData(dataset_raw):
    data = dataset_raw.data.to_frame().applymap(lambda x: list(x[:, :12].flatten()))
    data['label'] = dataset_raw.label.to_numpy()

    # Create label dictionaries (35 different known words)
    com_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']  # Command labels
    label_dict = dict(zip(com_labels, range(len(com_labels))))  # Commands oriented classification

    samples = np.array(data.data.tolist()).reshape((-1, 100, 12, 1))
    true_labels = data['label'].to_frame().applymap(lambda x: label_dict.get(x, len(label_dict))).to_numpy()
    return (data, samples, true_labels)

random.seed(3)


def main():
    test_dataset_raw = pd.read_hdf('dTest.h5')

    print('### Test data processing ###')
    data, samples, true_labels = extractData(test_dataset_raw)

    # model encoder
    print('###Model autoencoder initialization###')
    autoenc = tf.keras.models.load_model('autoencoders_models\\AUTOENCODER\\AutoencoderModel_withSCandBN150_train.h5')


    # model classifier
    print('###Model classifier initialization###')
    classifier = tf.keras.models.load_model('classifiers\\newDROPrate04_classifier_A_SCandBN_1204_1024_epochs200.h5')

    feat_out = autoenc.get_layer(name='feature_out').output
    in_x = autoenc.input
    encoder = Model(in_x, feat_out)

    print('###forwarding through encoder the dataset###')
    features = encoder.predict(samples)

    ### PREDICTION
    predicted_labels = classifier.predict(features)

    ### Confusion matrix
    # Predicted labels
    y_true = true_labels
    y_pred = np.argmax(predicted_labels, axis=1)
    # Evaluate confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Show the confusion matrix
    pd.DataFrame(cm)
    print('###results###')
    print(pd.DataFrame(cm).head(64))
    z = 1  # Linea di debugging


if __name__ == "__main__":
    main()