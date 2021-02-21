import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Reshape, Conv2DTranspose, ReLU, Cropping2D
from sklearn import decomposition, model_selection, metrics, manifold
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import datamerge
import pandas as pd

def extractData(dataset_raw):
    data = dataset_raw.data.to_frame().applymap(lambda x: list(x[:, :12].flatten()))
    data['label'] = dataset_raw.label.to_numpy()

    # Create label dictionaries (35 different known words)
    com_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']  # Command labels
    label_dict = dict(zip(com_labels, range(len(com_labels))))  # Commands oriented classification


    samples = data.data.tolist()
    samples = np.array(samples)
    samples = samples.reshape((-1, 100, 12, 1))

    true_labels = data['label'].to_frame().applymap(lambda x: label_dict.get(x, len(label_dict))).to_numpy()
    return(data, samples, true_labels)


def Classifier2L(input_shape, l1=1024, l2=2048):

    X_input = Input(input_shape)
    # Linear layers
    X = Dense(l1, activation='relu', name='linear0')(X_input)
    X = Dense(l2, activation='relu', name='linear1')(X)
    X = Dense(11, activation='softmax', name='out')(X)

    model = Model(inputs=X_input, outputs=X, name='Classifier')
    return model

def main():

    train_dataset_raw = datamerge.importDataset('dTrain2')
    #train_dataset_raw = pd.read_hdf('dVal.h5')

    print('### Train data processing ###')
    data, samples, true_labels = extractData(train_dataset_raw)

    print('### Validation data processing ###')
    val_dataset_raw = pd.read_hdf('dVal.h5')
    val_data, val_samples, val_labels = extractData(val_dataset_raw)


    hyparams = [2**i for i in range(9,12)]


    # Model
    print('Model initialization')
    autoenc = tf.keras.models.load_model('autoencoders_models\\AUTOENCODER\\AutoencoderModel_withSCandBN_train.h5')

    feat_out = autoenc.get_layer(name='feature_out').output
    in_x = autoenc.input
    encoder = Model(in_x, feat_out)
    # print(encoder.summary())

    features = encoder.predict(samples)
    val_features = encoder.predict(val_samples)

    n_features = 256

    train_name = 'train_loss'
    val_name = 'val_loss'
    train_acc = 'accuracy'
    val_acc = 'val_accuracy'
    for l2 in hyparams:
        for l1 in hyparams:

            print(f'### HYPERPARAMS l1: {l1} and l2: {l2} ###')
            predictor = Classifier2L(n_features, l1, l2)
            #print(predictor.summary())
            predictor.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            history = predictor.fit(x=features, y=true_labels, validation_data=(val_features, val_labels),
                                epochs=20, batch_size=256)
            with open("history\\GRIDSEARCH\\wSCandBatch\\history_"+train_name+"_"+str(l1)+"_"+str(l2)+".txt", "w") as output:
                output.write(str(history.history['loss']))
            with open("history\\GRIDSEARCH\\wSCandBatch\\history_"+val_name+"_"+str(l1)+"_"+str(l2)+".txt", "w") as output:
                output.write(str(history.history['val_loss']))
            with open("history\\GRIDSEARCH\\wSCandBatch\\history_" + train_acc + "_" + str(l1) + "_" + str(l2) + ".txt", "w") as output:
                output.write(str(history.history['accuracy']))
            with open("history\\GRIDSEARCH\\wSCandBatch\\history_" + val_acc + "_" + str(l1) + "_" + str(l2) + ".txt", "w") as output:
                output.write(str(history.history['val_accuracy']))


    z = 1  # Linea di debugging


if __name__ == "__main__":
    main()