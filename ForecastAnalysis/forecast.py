import pandas as pd
import datetime
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import glob
from PIL import Image
from natsort import natsorted, ns

# LUDWIG
# from ludwig.api import LudwigModel

# SKLEARN
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import max_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# KERAS
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

import warnings
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

test_size = 0.30

predictor = 600
epochs_nn = 300
epochs_lstm = 350
batch_size = 8

random = 12
n_jobs = 6
verbose = 2

def duration_model(file):
    features = pd.read_csv(file)

    infos(file, features)

    features = features.dropna()

    features['n'] = features['n'].astype('int32')

    labels = np.array(features['n'])
    mean = np.mean(labels)
    features = features.drop('n', axis=1)
    feature_list = list(features.columns)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=test_size, random_state=random, shuffle=True)

    ######################### MODEL DEFINITIONS ############################

    # model = RandomForestRegressor(n_estimators=predictor, random_state=random, verbose=1, n_jobs=n_jobs)
    model = GradientBoostingRegressor(n_estimators=predictor, random_state=random, verbose=0)
    # model = MLPRegressor(verbose=1)
    model.fit(train_features, train_labels)

    ######################### MODEL DEFINITIONS ############################
    predictions = model.predict(test_features)

    # plot(file, test_labels, predictions)
    importances(model, feature_list)
    errors(test_labels, predictions, mean)


def model_randomforest(file):
    features_basic = pd.read_csv(file)

    infos(file, features_basic)

    features = features_basic.copy()
    features = features.dropna()

    labels = np.array(features['n'])
    mean = np.mean(labels)
    features = features.drop('n', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=test_size, random_state=random, shuffle=False)

    print(train_labels.shape, test_labels.shape)

    ######################### MODEL DEFINITIONS ############################

    model = RandomForestRegressor(n_estimators=predictor, random_state=random, verbose=verbose, n_jobs=n_jobs)
    # model = GradientBoostingRegressor(n_estimators=predictor, random_state=random, verbose=0)
    model.fit(train_features, train_labels)

    ######################### MODEL DEFINITIONS ############################

    predictions = model.predict(test_features)
    all_predictions = model.predict(features)
    predictions = np.round(predictions, decimals=1)
    all_predictions = np.round(all_predictions, decimals=1)

    shift = int(file.split('-')[1])

    plot_mixed(file + '_RF', labels, all_predictions, shift)
    importances(model, feature_list)
    errors(test_labels, predictions, mean)
    return predictions


def model_keras_nn(file):
    features_basic = pd.read_csv(file)

    infos_nn(file, features_basic)

    features = features_basic.copy()
    features = features.dropna()

    labels = np.array(features['n'])
    mean = np.mean(labels)
    features = features.drop('n', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=test_size, random_state=random, shuffle=False)

    print(train_labels.shape, test_labels.shape)

    ######################### MODEL DEFINITIONS ############################

    estimator = KerasRegressor(build_fn=personal_model, shape=train_features.shape[1], epochs=epochs_nn, batch_size=batch_size, verbose=verbose)
    history = estimator.fit(train_features, train_labels)

    ######################### MODEL DEFINITIONS ############################

    # plot_history(file + '_NN_', history, 'loss', 'MAE')
    # plot_history(file + '_NN_', history, 'msle', 'MSLE')
    # plot_history(file + '_NN_', history, 'mse', 'MSE')

    predictions = estimator.predict(test_features)
    all_predictions = estimator.predict(features)
    predictions = np.round(predictions, decimals=1)
    all_predictions = np.round(all_predictions, decimals=1)

    shift = int((file.split('.')[0]).split('-')[2])

    # plot_predict(file + '_NN', test_labels, predictions, shift)
    # plot_mixed(file + '_NN', labels, all_predictions, shift)
    gif_plot2('No cross version', labels, all_predictions, shift)
    # weights(estimator, feature_list)
    errors(test_labels, predictions, mean)


def personal_model(shape):
    model = Sequential()
    model.add(Dense(shape, input_dim=shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(shape/2), activation='linear'))
    model.add(Dense(int(shape/4), activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam', metrics=['msle'])

    # model = Sequential()
    # model.add(Dense(shape*2, input_dim=shape, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(shape, activation='relu'))
    # model.add(Dense(int(shape/2), activation='relu'))
    # model.add(Dense(1, activation='linear'))
    # model.compile(loss='mae', optimizer='adam', metrics=['msle', 'mse'])
    # model.summary()

    return model


def model_keras_lstm(file):
    features_basic = pd.read_csv(file)

    infos_nn(file, features_basic)

    features = features_basic.copy()
    features = features.dropna()

    labels = np.array(features['n'])
    mean = np.mean(labels)
    features = features.drop('n', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=test_size, random_state=random, shuffle=False)

    train_features = train_features.reshape((train_features.shape[0], 1, train_features.shape[1]))
    test_features = test_features.reshape((test_features.shape[0], 1, test_features.shape[1]))

    print(train_labels.shape, test_labels.shape)

    ######################### MODEL DEFINITIONS ############################

    estimator = KerasRegressor(build_fn=lstm_model, shape=train_features.shape[2], epochs=epochs_lstm, batch_size=1, verbose=verbose)
    history = estimator.fit(train_features, train_labels)

    ######################### MODEL DEFINITIONS ############################

    # plot_history(file + '_NN_', history, 'loss', 'MSLE')
    # plot_history(file + '_NN_', history, 'mae', 'MAE')
    # plot_history(file + '_NN_', history, 'mse', 'MSE')

    predictions = estimator.predict(test_features)
    all_predictions = estimator.predict(features)
    predictions = np.round(predictions, decimals=0)

    plot_predict(file + '_LSTM', test_labels, predictions)
    plot_mixed(file + '_LSTM', labels, all_predictions)
    # weights(estimator, feature_list)
    errors(test_labels, predictions, mean)


def lstm_model(shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape, 1), activation='tanh', recurrent_activation='sigmoid'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam',  metrics=['mse', 'msle'])
    # model.summary()

    return model


def model_ludwig(v1, v2):
    fb1 = pd.read_csv(v1)
    ver1 = v1.split('_')[1]

    fb2 = pd.read_csv(v2)
    ver2 = v2.split('_')[1]

    infos_nn(v1, fb1)
    infos_nn(v2, fb2)

    f1 = fb1.copy()
    f1 = f1.dropna()

    f2 = fb2.copy()
    f2 = f2.dropna()

    l2 = np.array(f2['n'])
    mean = np.mean(l2)

    print(f1.shape, f2.shape)

    ######################### MODEL DEFINITIONS ############################

    model_definition = {...}
    ludwig_model = LudwigModel(model_definition, model_definition_file='data/model_definition.yaml')
    train_stats = ludwig_model.train(data_df=f1)

    ######################### MODEL DEFINITIONS ############################

    all_predictions = ludwig_model.predict(data_df=f2)
    all_predictions = np.round(all_predictions, decimals=1)

    shift = int((v2.split('.')[0]).split('-')[2])

    # plot_predict(file + '_NN', test_labels, predictions, shift)
    plot_mixed2('LUDWIG train: v{} - predict: v{} ==> {}'.format(ver1, ver2, shift), l2, all_predictions.values.reshape(all_predictions.shape[0]), shift)
    # weights(estimator, feature_list)
    errors(l2, all_predictions.values.reshape(all_predictions.shape[0]), mean)


def model_cross_version(vlist, v2):
    i = 0
    for v1 in vlist:
        fb1 = pd.read_csv(v1)
        ver1 = v1.split('_')[1]

        infos_nn(v1, fb1)

        f1 = fb1.copy()
        f1 = f1.dropna()

        fl1 = list(f1.columns)

        l1 = np.array(f1['n'])
        mean = np.mean(l1)
        f1 = f1.drop('n', axis=1)
        f1 = np.array(f1)

        ######################### MODEL DEFINITIONS ############################
        if i == 0:
            estimator = KerasRegressor(build_fn=personal_model, shape=f1.shape[1], epochs=epochs_nn, batch_size=batch_size, verbose=verbose)
        history = estimator.fit(f1, l1)

        ######################### MODEL DEFINITIONS ############################
        i += 1

    fb2 = pd.read_csv(v2)
    ver2 = v2.split('_')[1]

    infos_nn(v2, fb2)

    f2 = fb2.copy()
    f2 = f2.dropna()

    l2 = np.array(f2['n'])
    mean = np.mean(l2)
    f2 = f2.drop('n', axis=1)
    fl2 = list(f2.columns)
    f2 = np.array(f2)

    # plot_history(file + '_NN_', history, 'loss', 'MAE')
    # plot_history(file + '_NN_', history, 'msle', 'MSLE')
    # plot_history(file + '_NN_', history, 'mse', 'MSE')

    all_predictions = estimator.predict(f2)
    all_predictions = np.round(all_predictions, decimals=1)

    shift = int((v2.split('.')[0]).split('-')[2])

    # plot_predict(file + '_NN', test_labels, predictions, shift)
    gif_plot2('NORMAL train: v{} - predict: v{} ==> {}'.format(ver1, ver2, shift), estimator, f2, l2, shift)
    plot_mixed2('NORMAL train: v{} - predict: v{} ==> {}'.format(ver1, ver2, shift), l2, all_predictions, shift)
    # plot_mixed3('NORMAL train: v{} - predict: v{} ==> {}'.format(ver1, ver2, shift), l2, all_predictions, shift)
    # weights(estimator, feature_list)
    errors2(l2, all_predictions, mean, shift)


def model_recurrent(vlist, v2):

    i = 0
    for v1 in vlist:
        fb1 = pd.read_csv(v1)
        ver1 = v1.split('_')[1]

        infos_nn(v1, fb1)

        f1 = fb1.copy()
        f1 = f1.dropna()

        fl1 = list(f1.columns)

        col = ['severity_diff', 'n', 'mov_avg2', 'mov_avg4', '1before', '2before', '4before']

        # for c in col:
        #     x = f1[c].values.reshape(-1, 1)
        #     min_max_scaler = preprocessing.MinMaxScaler()
        #     x_scaled = min_max_scaler.fit_transform(x)
        #     f1[c] = pd.DataFrame(x_scaled)

        l1 = np.array(f1['n'])
        mean = np.mean(l1)
        f1 = f1.drop('n', axis=1)
        f1 = np.array(f1)

        f1 = f1.reshape((f1.shape[0], f1.shape[1], 1))

        ######################### MODEL DEFINITIONS ############################
        if i==0:
            estimator = KerasRegressor(build_fn=lstm_model, shape=f1.shape[1], epochs=epochs_lstm, batch_size=batch_size, verbose=verbose)
        history = estimator.fit(f1, l1)

        ######################### MODEL DEFINITIONS ############################
        i += 1

    fb2 = pd.read_csv(v2)
    ver2 = v2.split('_')[1]

    infos_nn(v2, fb2)

    f2 = fb2.copy()
    f2 = f2.dropna()
    fl2 = list(f2.columns)

    col = ['severity_diff', 'n', 'mov_avg2', 'mov_avg4', '1before', '2before', '4before']

    # for c in col:
    #     x = f2[c].values.reshape(-1, 1)
    #     min_max_scaler = preprocessing.MinMaxScaler()
    #     x_scaled = min_max_scaler.fit_transform(x)
    #     f2[c] = pd.DataFrame(x_scaled)

    l2 = np.array(f2['n'])
    mean = np.mean(l2)
    f2 = f2.drop('n', axis=1)
    f2 = np.array(f2)

    f2 = f2.reshape((f2.shape[0], f2.shape[1], 1))

    offset = 255

    lastf = f2[offset:offset+4]
    lastl = l2[offset:offset+4]

    for i in range(0, len(l2)):
        # p = estimator.predict(lastf[-1].reshape(1,lastf[-1].shape[0]))
        p = estimator.predict(lastf[-1].reshape(1, lastf[-1].shape[0], 1))
        # print(p)

        # CALCULATE NEW FEATURE FOR RECURRENT MODEL
        if lastf[-1][0] >= 52:
            w = 1
        else:
            w = lastf[-1][0] + 1

        diff = p - lastl[-1]
        mov2 = (lastl[-1] + lastl[-2]) / 2
        mov4 = (lastl[-1] + lastl[-2] + lastl[-3] + lastl[-4]) / 4
        b1 = lastl[-1]
        b2 = lastl[-2]
        b4 = lastl[-4]
        y = lastf[-1][7] + 1

        new_feature = np.array([int(w), int(diff), mov2, mov4, b1, b2, b4, int(y)])

        lastf = np.append(lastf, new_feature.reshape(1, new_feature.shape[0], 1), axis=0)
        lastl = np.append(lastl, p, axis=0)

    predictions = lastl[offset:]
    plot_predict2('train: v{} - predict: v{}'.format(ver1, ver2), l2[offset:], predictions[4:])

    # weights(estimator, feature_list)
    # errors(l2[50:], predictions, mean)


def infos(file, features_basic):
    print('#######################################')
    print('FILE:', file, '\n')
    print('test_size =', test_size)
    print('predictor = ', predictor)
    print('n_jobs = ', n_jobs)
    print('verbose = ', verbose)
    print('The shape of our features is:', features_basic.shape)


def infos_nn(file, features_basic):
    print('#######################################')
    print('FILE:', file, '\n')
    print('test_size =', test_size)
    print('epochs = ', epochs_nn)
    print('batch_size = ', batch_size)
    print('verbose = ', verbose)
    print('The shape of our features is:', features_basic.shape)


def plot_predict(file, test_labels, predictions, shift):
    n=[]
    for i in range(0, len(predictions)):
        n.append(i)

    plt.figure(figsize=(20, 11))
    sns.lineplot(n, test_labels, label='Real', ci=None)
    sns.lineplot(n[:-shift], predictions[shift:], label='Predict', ci=None)
    sns.lineplot(n, np.mean(test_labels), label='Mean', ci=None)
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('n')
    plt.minorticks_on()
    plt.grid(axis='both')
    plt.title('plot/' + file[5:] + '_predictions')
    # plt.savefig('plot/' + file + '_predictions.png', dpi=240)
    plt.show()


def plot_predict2(file, test_labels, predictions):
    n=[]
    for i in range(0, len(predictions)):
        n.append(i)

    plt.figure(figsize=(20, 11))
    sns.lineplot(n, test_labels, label='Real', ci=None)
    sns.lineplot(n, predictions, label='Predict', ci=None)
    sns.lineplot(n, np.mean(test_labels), label='Mean', ci=None)
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('n')
    plt.minorticks_on()
    plt.grid(axis='both')
    plt.title('plot/' + file[5:] + '_predictions')
    # plt.savefig('plot/' + file + '_predictions.png', dpi=240)
    plt.show()


def plot_mixed(file, labels, predictions, shift):
    n=[]
    for i in range(0, len(predictions)):
        n.append(i)

    plt.figure(figsize=(20, 11))
    sns.lineplot(n[:-shift], labels[:-shift], label='Real', ci=None)
    sns.lineplot(n[:-shift], predictions[shift:], label='Predict', ci=None)
    plt.axvline(int(len(predictions) * (1 - test_size)), linestyle='--', label='Split', c='red')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('n')
    plt.minorticks_on()
    plt.grid(axis='both')
    plt.title('plot/' + file[5:] + '_all_predictions')
    # plt.savefig('plot/' + file + '_all_predictions.png', dpi=240)
    plt.show()


def plot_mixed2(name, labels, predictions, shift):
    n=[]
    for i in range(0, len(predictions)):
        n.append(i)

    plt.figure(figsize=(20, 11))
    sns.lineplot(n[:-shift], labels[:-shift], label='Real', ci=None)
    sns.lineplot(n[:-shift], predictions[shift:], label='Predict', ci=None)
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('n')
    plt.minorticks_on()
    plt.grid(axis='both')
    plt.title(name)
    plt.savefig('plot/cross_version_' + str(shift) + '.png', dpi=240)
    plt.show()


def gif_plot(name, labels, predictions, shift):
    n=[]
    for i in range(0, len(predictions)):
        n.append(i)

    t0 = time.time()
    for i in range(3, len(predictions)):
        print('{}/{} [{} %]'.format(i, len(predictions), round(100*i/len(predictions), 1)))
        plt.figure(figsize=(10, 7))
        sns.lineplot(n[:-(len(predictions)-i)-shift], labels[:-(len(predictions)-i)-shift], label='Real', ci=None)
        sns.lineplot(n[:-(len(predictions)-i)-shift], predictions[shift:-(len(predictions)-i)], label='Predict', ci=None)
        plt.xticks(rotation='60')
        plt.legend()  # Graph labels
        plt.xlabel('Week')
        plt.ylabel('n')
        plt.minorticks_on()
        plt.grid(axis='both')
        plt.title(name + ' w#' + str(i))
        plt.savefig('plot/gif/cv_' + str(i) + '.png')
        # plt.show()
    print('Time elapsed: {} s'.format(round(time.time()-t0, 2)))

    print('Creating GIF...')

    # filepaths
    fp_in = 'plot/gif/cv_*.png'
    fp_out = 'plot/gif/cv.gif'

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in natsorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=35, loop=0, optimize=True)


def gif_plot2(name, estimator, test, labels, shift):
    n=[]
    for i in range(0, len(labels)):
        n.append(i)

    ymax = max(labels) + 70
    predictions = np.array([])

    try:
        t0 = time.time()
        for i in range(shift, len(labels)+shift, shift):
            if len(labels)-i<shift:
                diff = len(labels)-i
                i = i - diff

            print('{}/{} [{} %]'.format(i, len(labels), round(100*i/len(labels), 1)))
            plt.figure(figsize=(10, 7))
            plt.ylim(0, ymax)

            pred = estimator.predict(test[i:i+shift])

            predictions = np.append(predictions, pred)

            sns.lineplot(n[:i], labels[:i], label='Real', ci=None)
            # sns.lineplot(n[i:-(len(predictions)-i-shift)], labels[i:-(len(predictions)-i-shift)], label='Real*', ci=None, color='green')
            sns.lineplot(n[:i], predictions[:i], label='Predict*', ci=None, color='gainsboro')
            sns.lineplot(n[i-shift:i], pred, label='Predict', ci=None, color='orange')
            plt.xticks(rotation='60')
            plt.legend()  # Graph labels
            plt.xlabel('Week')
            plt.ylabel('n')
            plt.minorticks_on()
            plt.grid(axis='both')
            plt.title(name + ' w#' + str(i))
            plt.savefig('plot/gif/cv{}_{}.png'.format(shift, i))
            # plt.show()
    except:
        print('image error')

    print('Time elapsed: {} s'.format(round(time.time()-t0, 2)))
    print('Creating GIF...')

    # filepaths
    fp_in = 'plot/gif/cv{}_*.png'.format(shift)
    fp_out = 'plot/gif/cv{}.gif'.format(shift)

    img, *imgs = [Image.open(f) for f in natsorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=shift*40, loop=0, optimize=True)


def plot_mixed3(name, labels, predictions, shift):
    n = []
    for i in range(0, len(predictions)):
        n.append(i)

    plt.figure(figsize=(20, 11))
    sns.lineplot(n, labels, label='Real', ci=None)
    sns.lineplot(n, predictions, label='Predict', ci=None)
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('n')
    plt.minorticks_on()
    plt.grid(axis='both')
    plt.title(name)
    # plt.savefig('plot/' + file + '_all_predictions.png', dpi=240)
    plt.show()


def plot_history(file, history, name, label):
    plt.figure(figsize=(15, 8))
    sns.lineplot(history.epoch, history.history[name], label=label, ci=None)
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.minorticks_on()
    plt.grid(axis='both')
    plt.title('plot/' + file[5:] + label + '_History')
    plt.savefig('plot/' + file[5:] + label + '_history.png', dpi=240)
    plt.show()


def plot_all(df_original, file):
    plt.figure(figsize=(40, 18))
    df = df_original.copy()
    df['n'] = df['n'].astype('int32')
    df['date'] = df[['y', 'w']].astype(str).apply('-'.join, axis=1)
    sns.pointplot(df['date'], df['n'], label='value', ci=None, markersize=0.01, color='green')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('n')
    plt.grid(axis='both')
    plt.title('plot/Data Distribution: ' + file[5:])
    plt.savefig('plot/DataDistribution-' + file[5:] + '.png', dpi=240)
    plt.show()


def importances(model, feature_list):
    print('#######################################')
    # Get numerical feature importances
    importances = list(model.feature_importances_)  # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 4)) for feature, importance in
                           zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)
    [print('Variable: {:20} [{}]'.format(*pair)) for pair in feature_importances]


def errors(test_labels, predictions, mean):
    print('#######################################')
    # The baseline predictions are the historical averages
    baseline_errors = abs(mean - test_labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    index = []
    for i in range(0, len(test_labels)):
        if test_labels[i] == 0:
            index.append(i)

    test_labels = np.delete(test_labels, index)
    predictions = np.delete(predictions, index)


    errors = abs(predictions - test_labels)
    test_labels = test_labels[0:len(predictions)]

    MAE = mean_absolute_error(test_labels, predictions)
    EVS = explained_variance_score(test_labels, predictions)
    R2 = r2_score(test_labels, predictions)
    RSE = mean_squared_error(test_labels, predictions)
    REL = abs(100 * (abs(test_labels - predictions) / test_labels))
    MAX = max_error(test_labels, predictions)
    # MSLE = mean_squared_log_error(test_labels, predictions)

    print('Mean Absolute Error:', round(MAE, 2))
    # print('Max Error:', round(MAX, 2))
    # print('Exaplined Variance:', round(EVS, 3))
    print('R2 Scoring:', round(R2, 3))
    # print('MSLE Scoring:', round(MSLE, 5))
    # print('RSE:', round(RSE, 3))
    print('Relative:', np.round(np.mean(REL), 2), '%.')
    print('\nAccuracy:', np.round(100-np.mean(REL), 2), '%.')

    with open('errors.csv', 'a') as f:
        print(round(MAE, 2), round(R2, 3), np.round(np.mean(REL), 2), np.round(100-np.mean(REL), 2), sep=',', file=f)


def errors2(test_labels, predictions, mean, shift):
    print('#######################################')
    # The baseline predictions are the historical averages
    baseline_errors = abs(mean - test_labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    index = []
    for i in range(0, len(test_labels)):
        if test_labels[i] == 0:
            index.append(i)

    test_labels = np.delete(test_labels, index)
    predictions = np.delete(predictions, index)


    errors = abs(predictions - test_labels)
    test_labels = test_labels[0:len(predictions)]

    MAE = mean_absolute_error(test_labels, predictions)
    EVS = explained_variance_score(test_labels, predictions)
    R2 = r2_score(test_labels, predictions)
    RSE = mean_squared_error(test_labels, predictions)
    REL = abs(100 * (abs(test_labels - predictions) / test_labels))
    MAX = max_error(test_labels, predictions)
    # MSLE = mean_squared_log_error(test_labels, predictions)

    print('Relative:', np.round(np.mean(REL), 2), '%.')
    print('Accuracy:', np.round(100-np.mean(REL), 2), '%.')
    print('Mean Absolute Error:', round(MAE, 2))
    print('\nR2 Scoring:', round(R2, 3))

    with open('errors.csv', 'a') as f:
        print(shift, round(MAE, 2), round(R2, 3), np.round(np.mean(REL), 2), np.round(100-np.mean(REL), 2), sep=',', file=f)


def weights(estimator, list):
    for i, f in enumerate(list):
        print('{} \t\t {}'.format(f, estimator.model.weights[0][i]))