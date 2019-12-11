import pandas as pd
import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed

# LUDWIG
# from ludwig.api import LudwigModel

# SKLEARN
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

test_size = 0.25

predictor = 600
epochs_nn = 300
epochs_lstm = 500
batch_size = 8

random = 12
n_jobs = 6
verbose = 2

def duration_model(file):
    features = pd.read_csv(file + '.csv')

    infos(file, features)

    features = features.dropna()

    features['n'] = features['n'].astype('int32')

    labels = np.array(features['n'])
    mean = np.mean(labels)
    features = features.drop('n', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
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
    features_basic = pd.read_csv(file + '.csv')
    # plot_all(features_basic)

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

    ######################### MODEL DEFINITIONS ############################

    model = RandomForestRegressor(n_estimators=predictor, random_state=random, verbose=verbose, n_jobs=n_jobs)
    # model = GradientBoostingRegressor(n_estimators=predictor, random_state=random, verbose=0)
    model.fit(train_features, train_labels)

    ######################### MODEL DEFINITIONS ############################

    predictions = model.predict(test_features)
    all_predictions = model.predict(features)
    predictions = np.round(predictions, decimals=1)
    all_predictions = np.round(all_predictions, decimals=1)

    plot_predict(file + '_RF', test_labels, predictions)
    plot_mixed(file + '_RF', labels, all_predictions)
    importances(model, feature_list)
    errors(test_labels, predictions, mean)
    return predictions


def model_keras_nn(file):
    features_basic = pd.read_csv(file + '.csv')
    # plot_all(features_basic)

    infos_nn(file, features_basic)

    features = features_basic.copy()
    features = features.dropna()

    print(features.dtypes)

    labels = np.array(features['n'])
    mean = np.mean(labels)
    features = features.drop('n', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=test_size, random_state=random, shuffle=False)

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

    # plot_predict(file + '_NN', test_labels, predictions)
    plot_mixed(file + '_NN', labels, all_predictions)
    errors(test_labels, predictions, mean)


def personal_model(shape):
    model = Sequential()
    model.add(Dense(shape, input_dim=shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(int(shape/2), activation='relu'))
    model.add(Dense(int(shape/4), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam', metrics=['msle', 'mse'])
    # model.summary()

    return model


def model_keras_lstm(file):
    features_basic = pd.read_csv(file + '.csv')
    # plot_all(features_basic)

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
    errors(test_labels, predictions, mean)


def lstm_model(shape):
    model = Sequential()
    model.add(LSTM(shape, input_shape=(1, shape)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mae', optimizer='adam',  metrics=['mse', 'msle'])
    # model.summary()

    return model


def model_ludwig(file):
    features_basic = pd.read_csv(file + '.csv')

    infos_nn(file, features_basic)

    features = features_basic.copy()
    features = features.dropna()

    n = int(features.shape[0] * (1 - test_size))
    train = features.head(n)
    test = features.tail(features.shape[0] - n)

    ######################### MODEL DEFINITIONS ############################

    model_definition = {...}
    ludwig_model = LudwigModel(model_definition, model_definition_file='data/model_definition.yaml')
    train_stats = ludwig_model.train(data_df=train)

    ######################### MODEL DEFINITIONS ############################

    predictions = ludwig_model.predict(data_df=test, )
    all_predictions = ludwig_model.predict(data_df=features)
    predictions = np.round(predictions.n_predictions.values, decimals=1)
    all_predictions = np.round(all_predictions.n_predictions.values, decimals=1)

    plot_predict(file + '_LUDWIG', np.array(test['n']), np.array(predictions))
    plot_mixed(file + '_LUDWIG', np.array(features['n']), np.array(all_predictions))
    errors(np.array(test['n']), np.array(predictions), np.mean(features.n))


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


def plot_predict(file, test_labels, predictions):
    n=[]
    for i in range(0, len(predictions)):
        n.append(i)

    plt.figure(figsize=(15, 8))
    sns.lineplot(n, test_labels, label='real', ci=None)
    sns.lineplot(n, predictions, label='predict', ci=None)
    sns.lineplot(n, np.mean(test_labels), label='mean', ci=None)
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Issue')
    plt.ylabel('n')
    plt.minorticks_on()
    plt.grid(axis='both')
    plt.title(file[5:] + '_predictions')
    plt.savefig(file + '_predictions.png', dpi=240)
    plt.show()


def plot_mixed(file, labels, predictions):
    n=[]
    for i in range(0, len(predictions)):
        n.append(i)

    plt.figure(figsize=(15, 8))
    sns.lineplot(n, labels, label='real', ci=None)
    sns.lineplot(n, predictions, label='predict', ci=None)
    plt.axvline(int(len(predictions) * (1 - test_size)), linestyle='--', label='split', c='red')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('n')
    plt.minorticks_on()
    plt.grid(axis='both')
    plt.title(file[5:] + '_all_predictions')
    plt.savefig(file + '_all_predictions.png', dpi=240)
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
    plt.title(file[5:] + label + '_History')
    plt.savefig(file[5:] + label + '_history.png', dpi=240)
    plt.show()


def plot_all(df_original):
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
    plt.title('Data Distribution')
    plt.savefig('DataDistribution.png', dpi=240)
    plt.show()
    # exit(0)


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

    errors = abs(predictions - test_labels)
    test_labels = test_labels[0:len(predictions)]

    MAE = mean_absolute_error(test_labels, predictions)
    EVS = explained_variance_score(test_labels, predictions)
    R2 = r2_score(test_labels, predictions)
    RSE = mean_squared_error(test_labels, predictions)
    REL = 100 * (abs(test_labels - predictions) / test_labels)
    MAX = max_error(test_labels, predictions)
    # MSLE = mean_squared_log_error(test_labels, predictions)

    print('Mean Absolute Error:', round(MAE, 2))
    # print('Max Error:', round(MAX, 2))
    # print('Exaplined Variance:', round(EVS, 3))
    # print('R2 Scoring:', round(R2, 3))
    # print('MSLE Scoring:', round(MSLE, 5))
    # print('RSE:', round(RSE, 3))
    print('Relative:', np.round(np.mean(REL), 2), '%.')
    print('\nAccuracy:', np.round(100-np.mean(REL), 2), '%.')