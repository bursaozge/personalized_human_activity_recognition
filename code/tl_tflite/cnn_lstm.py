# Last info
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

from tensorflow import keras

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import layers

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder

# import cProfile, pstats
# from hurry.filesize import size
import os
# import psutil
# import wmi

from sklearn.metrics import accuracy_score, confusion_matrix

### Both transfer learning and tflite were used with Mobiact data.

# import seaborn as sns
# from pylab import rcParams
# import matplotlib.pyplot as plt
# from matplotlib import rc
# from pandas.plotting import register_matplotlib_converters
from numpy import save, load
from datetime import datetime


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


def converter_tflite(model):
    return tf.lite.TFLiteConverter.from_keras_model(model)


TFLITE_MODEL = 'models/mobiact/tl_tflite/cnn_lstm_lite_{}_{}.tflite'  # model type, user
TFLITE_DYNAMIC_MODEL = 'models/mobiact/tl_tflite/cnn_lstm_dynamic_{}_{}.tflite'
TFLITE_FLOAT16_MODEL = 'models/mobiact/tl_tflite/cnn_lstm_float16_{}_{}.tflite'


def convert_tflite(model, tflite_type, model_type, user):
    MODEL_TYPE_USER = None

    if tflite_type == 'lite':
        MODEL_TYPE_USER = TFLITE_MODEL.format(model_type, user)
        converter = converter_tflite(model)

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]

        tflite_model = converter.convert()

    elif tflite_type == 'dynamic':
        MODEL_TYPE_USER = TFLITE_DYNAMIC_MODEL.format(model_type, user)

        # Convert the model
        # Dynamic
        converter = converter_tflite(model)  # path to the SavedModel directory
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()

    elif tflite_type == 'float16':
        MODEL_TYPE_USER = TFLITE_FLOAT16_MODEL.format(model_type, user)

        # Convert the model
        # Float16
        converter = converter_tflite(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()

    # Save the model.
    if MODEL_TYPE_USER is not None:
        with open(MODEL_TYPE_USER, 'wb') as f:
            f.write(tflite_model)


def load_tflite(tflite_type, model_type, user, X_test, y_test):
    MODEL_TYPE_USER = None

    if tflite_type == 'lite':
        MODEL_TYPE_USER = TFLITE_MODEL.format(model_type, user)

    elif tflite_type == 'dynamic':
        MODEL_TYPE_USER = TFLITE_DYNAMIC_MODEL.format(model_type, user)

    elif tflite_type == 'float16':
        MODEL_TYPE_USER = TFLITE_FLOAT16_MODEL.format(model_type, user)

    ####### TFLITE #######
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=MODEL_TYPE_USER, num_threads=2)  # (model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]

    input_index = input_det["index"]
    output_index = output_det["index"]
    input_shape = input_det["shape"]
    output_shape = output_det["shape"]
    input_dtype = input_det["dtype"]
    output_dtype = output_det["dtype"]

    def predict(inp, input_dtype, output_shape, output_dtype, interpreter, input_index, output_index):
        inp = inp.astype(input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, output_shape[1]), dtype=output_dtype)

        interpreter.resize_tensor_input(input_index, [1, 4, 20, 3])
        interpreter.allocate_tensors()

        for i in range(count):
            interpreter.set_tensor(input_index, inp[i:i + 1])
            interpreter.invoke()
            out[i] = interpreter.get_tensor(output_index)[0]
        return out

    start_time_tflite = datetime.now()
    y_pred_lite = predict(X_test[:1], input_dtype, output_shape, output_dtype, interpreter, input_index, output_index)
    print('TFLite time: ', datetime.now() - start_time_tflite)
    y_pred_lite = predict(X_test, input_dtype, output_shape, output_dtype, interpreter, input_index, output_index)

    y_pred_lite = np.argmax(y_pred_lite, axis=1)
    y_test_lite = np.argmax(y_test, axis=1)

    acc_lite = accuracy_score(y_test_lite, y_pred_lite)

    return acc_lite


def tflite(model, model_type, user, X_test, y_test):

    for tflite_type in ['lite', 'dynamic', 'float16']:
        convert_tflite(model, tflite_type, model_type, user)

    acc_lite = load_tflite('lite', model_type, user, X_test, y_test)
    acc_lite_dynamic = load_tflite('dynamic', model_type, user, X_test, y_test)
    acc_lite_float16 = load_tflite('float16', model_type, user, X_test, y_test)

    return acc_lite, acc_lite_dynamic, acc_lite_float16


def main():
    data_org = load_data()

    users = data_org.User.value_counts().index

    for user in users:
        results = pd.DataFrame(columns=['User', 'General_Acc', 'Model_Acc', 'New_Model_Acc', 'Support', 'TF_Format'])

        data_test = data_org[data_org.User == user]
        user_shape = data_test.shape[0]

        data = data_org[data_org.User != user]

        df = data.drop(['User', 'Time'], axis=1)
        df_test = data_test.drop(['User', 'Time'], axis=1)

        print('df: ', df.shape, 'df_test: ', df_test.shape)

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

        X_train_path = f"data/MobiAct/users/20train/X_train_{user}.npy"
        y_train_path = f"data/MobiAct/users/20train/y_train_{user}.npy"
        X_test_path = f"data/MobiAct/users/20train/X_test_{user}.npy"
        y_test_path = f"data/MobiAct/users/20train/y_test_{user}.npy"

        if not os.path.isfile(X_train_path):

            X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.8, random_state=42, stratify=y_t)
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

        verbose, epochs, batch_size = 0, 20, 64
        n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]

        # reshape
        n_steps, n_length = 4, 20
        X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
        # X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_features))

        X_train_t = X_train_t.reshape((X_train_t.shape[0], n_steps, n_length, n_features))
        X_test_t = X_test_t.reshape((X_test_t.shape[0], n_steps, n_length, n_features))

        ##################################BASE MODEL##################################

        model_path = 'models/mobiact/tl/base_model_{}.h5'.format(user)

        if os.path.exists(model_path):
            print('Model is already')
            model = tf.keras.models.load_model(model_path)

        else:

            model = Sequential()
            model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                                      input_shape=(None, n_length, n_features)))
            model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
            model.add(TimeDistributed(Dropout(0.5)))
            model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
            model.add(TimeDistributed(Flatten()))
            model.add(LSTM(100))
            model.add(Dropout(0.5))
            model.add(Dense(100, activation='relu'))
            model.add(Dense(n_outputs, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # fit network
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

            model.save(model_path)

        y_pred_base = model.predict(X_test_t)

        y_pred_base = np.argmax(y_pred_base, axis=1)
        y_test_base = np.argmax(y_test_t, axis=1)

        acc_base = accuracy_score(y_test_base, y_pred_base)

        acc_lite_base, acc_lite_dynamic_base, acc_lite_float16_base = tflite(model, 'base', user, X_test_t, y_test_t)

        ##################################TL MODEL##################################

        new_model = tf.keras.models.clone_model(model)
        new_model.set_weights(model.get_weights())

        new_model.layers.pop()
        print(len(new_model.layers))

        new_model.outputs = [new_model.layers[-1].output]

        # model.layers[-1].outbound_nodes = []

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

        x = Dense(y_train_t.shape[1], activation='softmax', name="dense_b")(x)

        new_model = Model(inputs=new_model.input, outputs=x)

        new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        new_model.fit(X_train_t, y_train_t, epochs=100, batch_size=32, verbose=1)

        y_pred_tl = new_model.predict(X_test_t)

        y_pred_tl = np.argmax(y_pred_tl, axis=1)
        y_test_tl = np.argmax(y_test_t, axis=1)

        acc_tl = accuracy_score(y_test_tl, y_pred_tl)

        acc_lite_tl, acc_lite_dynamic_tl, acc_lite_float16_tl = tflite(new_model, 'tl', user, X_test_t, y_test_t)

        ##################################GENERAL MODEL##################################

        model_t = Sequential()
        model_t.add(
            TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features)))
        model_t.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model_t.add(TimeDistributed(Dropout(0.5)))
        model_t.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model_t.add(TimeDistributed(Flatten()))
        model_t.add(LSTM(100))
        model_t.add(Dropout(0.5))
        model_t.add(Dense(100, activation='relu'))
        model_t.add(Dense(n_outputs, activation='softmax'))
        model_t.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model_t.fit(X_train_t, y_train_t, epochs=epochs, batch_size=batch_size, verbose=1)

        y_pred_general = model_t.predict(X_test_t)

        y_pred_general = np.argmax(y_pred_general, axis=1)
        y_test_general = np.argmax(y_test_t, axis=1)

        acc_general = accuracy_score(y_test_general, y_pred_general)

        acc_lite_general, acc_lite_dynamic_general, acc_lite_float16_general = tflite(model_t, 'general', user, X_test_t, y_test_t)

        #cnn_tl_results_file = 'result/basemodel:mobiact_test:mobiact/results_cnn_lstm.csv'
        cnn_tl_results_file = 'results/experiment10/results_cnn_lstm_20train_2062022.csv'

        results = results.append({'User': user, 'General_Acc': acc_general, 'Model_Acc': acc_base, 'New_Model_Acc': acc_tl,
                                  'Support': user_shape, 'TF_Format': 'TF'}, ignore_index=True)

        results = results.append({'User': user, 'General_Acc': acc_lite_general, 'Model_Acc': acc_lite_base,
                                  'New_Model_Acc': acc_lite_tl, 'Support': user_shape, 'TF_Format': 'TF Lite'},
                                 ignore_index=True)

        results = results.append({'User': user, 'General_Acc': acc_lite_dynamic_general, 'Model_Acc': acc_lite_dynamic_base,
                                  'New_Model_Acc': acc_lite_dynamic_tl, 'Support': user_shape, 'TF_Format': 'TF Lite + Dynamic'},
                                 ignore_index=True)

        results = results.append({'User': user, 'General_Acc': acc_lite_float16_general, 'Model_Acc': acc_lite_float16_base,
                                  'New_Model_Acc': acc_lite_float16_tl, 'Support': user_shape, 'TF_Format': 'TF Lite + Float16'},
                                 ignore_index=True)

        results = round(results, 3)

        if not os.path.isfile(cnn_tl_results_file):
            results.to_csv(cnn_tl_results_file, index=False)
        else:
            results.to_csv(cnn_tl_results_file, mode='a', header=False, index=False)

        del model
        del new_model
        del model_t


if __name__ == '__main__':
    main()