# Last info
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, LSTM, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

from tensorflow import keras

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D 
from tensorflow.keras import layers

import pandas as pd
import numpy as np
from datetime import datetime
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder

#import cProfile, pstats
#from hurry.filesize import size
import os
#import psutil
#import wmi

from sklearn.metrics import accuracy_score, confusion_matrix
from numpy import save, load

#import seaborn as sns
#from pylab import rcParams
#import matplotlib.pyplot as plt
#from matplotlib import rc
#from pandas.plotting import register_matplotlib_converters

##### Mobiact veri ile userlar test edilmiştir.


N_STEPS, N_LENGTH = 4, 20
VERBOSE, EPOCHS, BATCH_SIZE = 0, 20, 64
N_FEATURES = 3
N_OUTPUTS = None
RESULTS_FILE = 'results/experiment6/results_convlstm2d.csv'


def load_data(data_path='data/MobiAct/raw_data_without_CHU_SIT.csv'):
    data = pd.read_csv(data_path)

    data['x'] = data['x'].astype('float')
    data['y'] = data['y'].astype('float')
    data['z'] = data['z'].astype('float')
    
    return data


def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels


def create_dir(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def save_file(data, file):
    create_dir(file)
    if not os.path.isfile(file):
        data.to_csv(file, index=False)
    else:
        data.to_csv(file, mode='a', header=False, index=False)


def preprocess_data(df, df_test, user):
    # Label Encoder
    label = LabelEncoder()
    df['label'] = label.fit_transform(df['Activity'])
    df_test['label'] = label.transform(df_test['Activity'])

    # Standardrizing
    X = df[['x', 'y', 'z']]
    y = df['label']

    X_t = df_test[['x', 'y', 'z']]
    y_t = df_test['label']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    scaled_X = pd.DataFrame(data=X, columns=['x', 'y', 'z'])
    scaled_X['label'] = y.values

    X_t = scaler.transform(X_t)
    scaled_X_t = pd.DataFrame(data=X_t, columns=['x', 'y', 'z'])
    scaled_X_t['label'] = y_t.values

    Fs = 20
    frame_size = Fs * 4  # 80
    hop_size = Fs * 2  # 40

    X, y = get_frames(scaled_X, frame_size, hop_size)
    X_t, y_t = get_frames(scaled_X_t, frame_size, hop_size)

    X_train = X;
    y_train = y;
    # X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size = 0.6, random_state = 42, stratify = y_t)
    # aynıları kullanılması için bunlar kaydedilecek.

    X_train_path = f"data/MobiAct/users/X_train_{user}.npy"
    y_train_path = f"data/MobiAct/users/y_train_{user}.npy"
    X_test_path = f"data/MobiAct/users/X_test_{user}.npy"
    y_test_path = f"data/MobiAct/users/y_test_{user}.npy"

    if not os.path.isfile(X_train_path):
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.6, random_state=42,
                                                                    stratify=y_t)

        save(X_train_path, X_train_t)
        save(y_train_path, y_train_t)
        save(X_test_path, X_test_t)
        save(y_test_path, y_test_t)

        '''
        # to save it for train
        with open(train_path, "w") as f:
            pkl.dump([X_train_t, y_train_t], f)

        # to save it for test
        with open(test_path, "w") as f:
            pkl.dump([X_test_t, y_test_t], f)
        '''
    else:
        X_train_t = load(X_train_path)
        y_train_t = load(y_train_path)
        X_test_t = load(X_test_path)
        y_test_t = load(y_test_path)

        '''
        # to load it for train
        with open(train_path, "r") as f:
            X_train_t, y_train_t = pkl.load(f)

        # to load it for test
        with open(test_path, "r") as f:
            X_test_t, y_test_t = pkl.load(f)
        '''

    print(X_train.shape, y_train.shape)
    print(X_train_t.shape, X_test_t.shape)

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

    enc = enc.fit(y_train.reshape(-1, 1))

    y_train = enc.transform(y_train.reshape(-1, 1))
    # y_test = enc.transform(y_test.reshape(-1, 1))

    y_train_t = enc.transform(y_train_t.reshape(-1, 1))
    y_test_t = enc.transform(y_test_t.reshape(-1, 1))

    global N_FEATURES, N_OUTPUTS
    #verbose, epochs, batch_size = 0, 20, 64
    n_timesteps, N_FEATURES, N_OUTPUTS = X_train.shape[1], X_train.shape[2], y_train.shape[1]

    # reshape
    # n_steps, n_length = 4, 20
    X_train = X_train.reshape((X_train.shape[0], N_STEPS, 1, N_LENGTH, N_FEATURES))
    # X_test = X_test.reshape((X_test.shape[0], N_STEPS, N_LENGTH, N_FEATURES))

    X_train_t = X_train_t.reshape((X_train_t.shape[0], N_STEPS, 1, N_LENGTH, N_FEATURES))
    X_test_t = X_test_t.reshape((X_test_t.shape[0], N_STEPS, 1, N_LENGTH, N_FEATURES))

    return X_train, y_train, X_train_t, y_train_t, X_test_t, y_test_t


def main():
    data_org = load_data()

    users = data_org.User.value_counts().index

    for user in users:

        print('User: ', user)

        results = pd.DataFrame(columns=['User', 'General_Acc', 'Model_Acc', 'New_Model_Acc', 'Support'])

        data_test = data_org[data_org.User == user]
        user_shape = data_test.shape[0]

        data = data_org[data_org.User != user]

        df = data.drop(['User', 'Time'],axis=1)
        df_test = data_test.drop(['User', 'Time'],axis=1)

        print('df: ', df.shape, 'df_test: ', df_test.shape)

        X_train, y_train, X_train_t, y_train_t, X_test_t, y_test_t = preprocess_data(df, df_test, user)

        ##################################BASE MODEL##################################

        model_path = 'models/mobiact/convlstm2d/base_model_{}.h5'.format(user)
        start_time_base = datetime.now()
        if os.path.exists(model_path):
            print('Model is already')
            model = tf.keras.models.load_model(model_path)

        else:

            model = Sequential()
            model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3),
                                 activation='relu'))  # input_shape=(n_steps, 1, n_length, n_features)))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(100, activation='relu'))
            model.add(Dense(N_OUTPUTS, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # fit network
            model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

            model.save(model_path)
        print('Base %40 train time: ', datetime.now() - start_time_base)

        start_time_basep = datetime.now()
        y_pred_base = model.predict(X_test_t)
        print('Base predict time: ', datetime.now() - start_time_basep)

        y_pred_base = np.argmax(y_pred_base, axis=1)
        y_test_base = np.argmax(y_test_t, axis=1)

        acc_base = accuracy_score(y_test_base, y_pred_base)

        ##################################TL MODEL##################################
        start_time_tl = datetime.now()

        new_model = tf.keras.models.clone_model(model)
        new_model.set_weights(model.get_weights())

        new_model.layers.pop()
        print(len(new_model.layers))

        new_model.outputs = [new_model.layers[-1].output]

        #model.layers[-1].outbound_nodes = []

        for layer in new_model.layers[:-2]:
            layer.trainable = False

        for layer in new_model.layers[-2:]:
            layer.trainable = True

        print('y_shape: ', y_train_t.shape[1])

        x = layers.Dense(100, name="dense_f")(new_model.output)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        # Add a dropout rate of 0.5
        x = layers.Dropout(0.5, name="do_f")(x)
        # One more dense layer
        x = layers.Dense(100, name="dense_g")(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5, name="do_g")(x)
        x = layers.BatchNormalization()(x)

        x=Dense(y_train_t.shape[1], activation='softmax', name="dense_b")(x)

        new_model = Model(inputs=new_model.input,outputs=x)

        new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        new_model.fit(X_train_t, y_train_t, epochs=100, batch_size=32, verbose=1)
        print('TL train time: ', datetime.now() - start_time_tl)

        start_time_tlp = datetime.now()
        y_pred_tl = new_model.predict(X_test_t)
        print('TL predict time: ', datetime.now() - start_time_tlp)

        y_pred_tl = np.argmax(y_pred_tl, axis=1)
        y_test_tl = np.argmax(y_test_t, axis=1)

        acc_tl = accuracy_score(y_test_tl, y_pred_tl)

        ##################################GENERAL MODEL##################################
        start_time_g = datetime.now()

        model_t = Sequential()
        model_t.add(ConvLSTM2D(filters=64, kernel_size=(1, 3),
                             activation='relu'))  # input_shape=(n_steps, 1, n_length, n_features)))
        model_t.add(Dropout(0.5))
        model_t.add(Flatten())
        model_t.add(Dense(100, activation='relu'))
        model_t.add(Dense(N_OUTPUTS, activation='softmax'))
        model_t.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit network
        model_t.fit(X_train_t, y_train_t, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        print('General train time: ', datetime.now() - start_time_g)

        start_time_gp = datetime.now()
        y_pred_general = model_t.predict(X_test_t)
        print('General predict time: ', datetime.now() - start_time_gp)

        y_pred_general = np.argmax(y_pred_general, axis=1)
        y_test_general = np.argmax(y_test_t, axis=1)

        acc_general = accuracy_score(y_test_general, y_pred_general)

        results = results.append({'User': user, 'General_Acc': acc_general , 'Model_Acc': acc_base,
                                  'New_Model_Acc': acc_tl, 'Support': user_shape}, ignore_index=True)

        results = round(results, 3)

        save_file(results, RESULTS_FILE)

        del model
        del new_model
        del model_t


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))