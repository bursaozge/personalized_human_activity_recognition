import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow import keras

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import layers

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder

import os
from sklearn.metrics import recall_score, precision_score, f1_score

### Openhar verisi kullanılarak farklı veri setleri üzerinden yeni class predicti değerlendirildi.

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


org_path = "dataset_10_position_4" #"dataset_10_position_4" #"without_dataset_10"
test_path = "dataset_9_position_9" #"dataset_9_position_8" #"dataset_10_position_4"

# data_org = load_data('data/MobiAct/raw_data_without_CHU_SIT.csv')
#data_org = load_data('data/OpenHAR/processed2/{}.csv'.format(org_path))
data_org = load_data('data/OpenHAR/processed_merge/{}.csv'.format(org_path))

#data_test_org = load_data('data/OpenHAR/processed2/{}.csv'.format(test_path))
data_test_org = load_data('data/OpenHAR/processed_merge/{}.csv'.format(test_path))

users = data_test_org.User.value_counts().index
print(users)

for user in users:

    print('User: ', user)

    try:
        results = pd.DataFrame(columns=['User', 'General_Acc', 'Model_Acc', 'New_Model_Acc', 'Support'])

        data_test = data_test_org[data_test_org.User == user]
        user_shape = data_test.shape[0]

        data = data_org.copy()

        df = data.drop(['User', 'Time'], axis=1)
        df_test = data_test.drop(['User', 'Time'], axis=1)

        print('df: ', df.shape, 'df_test: ', df_test.shape)

        # Label Encoder
        label = LabelEncoder()
        tmp = df.append(df_test)
        tmp['label'] = label.fit_transform(tmp['Activity'])

        df['label'] = label.transform(df['Activity'])
        df_test['label'] = label.transform(df_test['Activity'])

        tmp_label = tmp[['Activity', 'label']].drop_duplicates()
        df_label = df[['Activity', 'label']].drop_duplicates()
        df_test_label = df_test[['Activity', 'label']].drop_duplicates()

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
        X_t, y_t = get_frames(scaled_X_t, frame_size, hop_size)  # bölü 20

        X_train = X;
        y_train = y;
        #X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.6, random_state=42,
        # stratify=y_t)
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_t, y_t, test_size=0.8, random_state=42,
                                                                    stratify=y_t)

        print(X_train.shape, y_train.shape)
        print(X_train_t.shape, X_test_t.shape)

        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

        # enc = enc.fit(y_train.reshape(-1, 1))
        enc = enc.fit(np.append(y_train, np.append(y_train_t, y_test_t)).reshape(-1, 1))

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

        X_train_t = X_train_t.reshape((X_train_t.shape[0], n_steps, n_length, n_features))  # bölü 20
        X_test_t = X_test_t.reshape((X_test_t.shape[0], n_steps, n_length, n_features))

        ##################################BASE MODEL##################################

        # model_path = 'model/openhar/base_model_all_users_new_label.h5'
        # model_path = 'models/openhar-newmobiact/base_model_{}_trousers.h5'.format(org_path)
        model_path = 'models/openhar-newmobiact_20train_merge/base_model_{}.h5'.format(org_path)

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
        # labels=df_label.sort_values('label').Activity.values
        # Recall
        recall_base = recall_score(y_test_base, y_pred_base, average=None)
        # Precision
        precision_base = precision_score(y_test_base, y_pred_base, average=None)
        # F1 score
        f1_score_base = f1_score(y_test_base, y_pred_base, average=None)

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

        print('TL MODEL LAST DENSE SENSE : ', y_train_t.shape[1])
        x = Dense(y_train_t.shape[1], activation='softmax', name="dense_b")(x)

        new_model = Model(inputs=new_model.input, outputs=x)

        new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        new_model.fit(X_train_t, y_train_t, epochs=100, batch_size=32, verbose=1)

        y_pred_tl = new_model.predict(X_test_t)

        y_pred_tl = np.argmax(y_pred_tl, axis=1)
        y_test_tl = np.argmax(y_test_t, axis=1)

        acc_tl = accuracy_score(y_test_tl, y_pred_tl)

        # Recall
        recall_tl = recall_score(y_test_tl, y_pred_tl, average=None)
        # Precision
        precision_tl = precision_score(y_test_tl, y_pred_tl, average=None)
        # F1 score
        f1_score_tl = f1_score(y_test_tl, y_pred_tl, average=None)

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
        # model_t.add(Dense(n_outputs, activation='softmax'))
        model_t.add(Dense(y_train_t.shape[1], activation='softmax'))
        model_t.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        model_t.fit(X_train_t, y_train_t, epochs=epochs, batch_size=batch_size, verbose=1)

        y_pred_general = model_t.predict(X_test_t)

        y_pred_general = np.argmax(y_pred_general, axis=1)
        y_test_general = np.argmax(y_test_t, axis=1)

        acc_general = accuracy_score(y_test_general, y_pred_general)

        # Recall
        recall_general = recall_score(y_test_general, y_pred_general, average=None)
        # Precision
        precision_general = precision_score(y_test_general, y_pred_general, average=None)
        # F1 score
        f1_score_general = f1_score(y_test_general, y_pred_general, average=None)

        # cnn_tl_results_file = 'result/openhar/results_cnn_lstm_new_class_dataset_5_position_1_22022022.csv'
        cnn_tl_results_file = 'results/experiment8_20train_merge/{}/results_cnn_lstm_new_class_31052022.csv'.format(test_path)
        # cnn_tl_results_cm_base_file = 'result/results_cnn_lstm_new_class_cm_base_15022022.csv'
        cnn_tl_results_cm_new_file = 'results/experiment8_20train_merge/{}/results_cnn_lstm_new_class_cm_31052022.csv'.format(
            test_path)

        results = results.append({'User': user, 'General_Acc': acc_general, 'Model_Acc': acc_base, 'New_Model_Acc': acc_tl,
                                  'Support': user_shape}, ignore_index=True)
        '''
        results_cm = pd.DataFrame([recall_general, recall_base, recall_tl,
                                   precision_general, precision_base, precision_tl,
                                   f1_score_general, f1_score_base, f1_score_tl],
                                  columns=label.classes_)
        results_cm['Type'] = ['Recall General', 'Recall Base', 'Recall TL',
                              'Predicision General', 'Precision Base', 'Precision TL',
                              'F1 Score General', 'F1 Score Base', 'F1 Score TL']
        '''

        '''
        ############## BASE ###############################
        results_cm_base = pd.DataFrame([ recall_base,
                                    precision_base,
                                    f1_score_base],
                                  columns=df_label.sort_values('label').Activity.values)
        results_cm_base['Type'] = [ 'Recall Base',
                                    'Precision Base',
                                    'F1 Score Base']
        results_cm_base['User'] = user
        '''

        #''' 
        ############## NEW CLASS ###############################
        results_cm_new = pd.DataFrame([recall_general, recall_tl,
                                       precision_general, precision_tl,
                                       f1_score_general, f1_score_tl],
                                      columns=df_test_label.sort_values('label').Activity.values)
        results_cm_new['Type'] = ['Recall General', 'Recall TL',
                                  'Predicision General', 'Precision TL',
                                  'F1 Score General', 'F1 Score TL']
        results_cm_new['User'] = user
    
        # results_cm_base = round(results_cm_base, 3)
        results_cm_new = round(results_cm_new, 3)
        #'''
        results = round(results, 3)

        if not os.path.isfile(cnn_tl_results_file):
            results.to_csv(cnn_tl_results_file, index=False)
        else:
            results.to_csv(cnn_tl_results_file, mode='a', header=False, index=False)

        '''
        if not os.path.isfile(cnn_tl_results_cm_base_file):
            results_cm_base.to_csv(cnn_tl_results_cm_base_file, index=False)
        else:
            results_cm_base.to_csv(cnn_tl_results_cm_base_file, mode='a', header=False, index=False)
        '''
        #'''
        if not os.path.isfile(cnn_tl_results_cm_new_file):
            results_cm_new.to_csv(cnn_tl_results_cm_new_file, index=False)
        else:
            results_cm_new.to_csv(cnn_tl_results_cm_new_file, mode='a', header=False, index=False)
        #'''
        del model
        del new_model
        del model_t

    except Exception as e:
        print(e)
