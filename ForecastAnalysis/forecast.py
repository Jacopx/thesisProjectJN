import pandas as pd
import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed

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
epochs_lstm = 100
batch_size = 4

random = 12
n_jobs = 6
verbose = 0

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


def count_model(file):
    features_basic = pd.read_csv(file + '.csv')
    plot_all(features_basic)

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
    predictions = np.round(predictions, decimals=0)

    plot(file + ' RF', test_labels, predictions)
    # importances(model, feature_list)
    errors(test_labels, predictions, mean)
    return predictions


def count_model_keras_nn(file):
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

    estimator = KerasRegressor(build_fn=personal_model, shape=train_features.shape[1], epochs=epochs_nn, batch_size=batch_size, verbose=verbose)
    estimator.fit(train_features, train_labels)

    ######################### MODEL DEFINITIONS ############################

    predictions = estimator.predict(test_features)
    predictions = np.round(predictions, decimals=0)

    plot(file + ' NN', test_labels, predictions)
    # importances(model, feature_list)
    errors(test_labels, predictions, mean)


def personal_model(shape):
    model = Sequential()
    model.add(Dense(shape, input_dim=shape, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(shape, activation='relu'))
    model.add(Dense(int(shape/2), activation='relu'))
    # model.add(Dense(36, activation='linear'))
    # model.add(Dense(28, activation='relu'))
    model.add(Dense(int(shape/4), activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mae'])
    # model.summary()

    return model


def count_model_keras_lstm(file):
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

    train_features = train_features.reshape((train_features.shape[0], 1, train_features.shape[1]))
    test_features = test_features.reshape((test_features.shape[0], 1, test_features.shape[1]))

    ######################### MODEL DEFINITIONS ############################

    estimator = KerasRegressor(build_fn=lstm_model, shape=train_features.shape[1], epochs=epochs_lstm, batch_size=1, verbose=verbose)
    estimator.fit(train_features, train_labels)

    ######################### MODEL DEFINITIONS ############################

    predictions = estimator.predict(test_features)
    predictions = np.round(predictions, decimals=0)

    plot(file + ' LSTM', test_labels, predictions)
    # importances(model, feature_list)
    errors(test_labels, predictions, mean)


def lstm_model(shape):
    model = Sequential()
    model.add(LSTM(shape, input_shape=(1, shape)))
    model.add(Dense(63, activation='relu'))
    # model.add(Dense(36, activation='linear'))
    # model.add(Dense(28, activation='relu'))
    # model.add(Dense(12, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam',  metrics=['accuracy', 'mae'])
    # model.summary()

    return model


def infos(file, features_basic):
    print('#######################################')
    print('FILE:', file, '\n')
    print('test_size =', test_size)
    print('predictor = ', predictor)
    print('n_jobs = ', n_jobs)
    print('The shape of our features is:', features_basic.shape)


def plot(file, test_labels, predictions):
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
    plt.title(file + ' predictions')
    plt.savefig(file + '_predictions.png', dpi=240)
    plt.show()


def plot_all(df_original):
    plt.figure(figsize=(40, 18))
    df = df_original.copy()
    df['n'] = df['n'].astype('int32')
    df['date'] = df[['y', 'w']].astype(str).apply('-'.join, axis=1)
    sns.pointplot(df['date'], df['n'], label='value', ci=None, markersize=0.01, color='green')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Date')
    plt.ylabel('n')
    # plt.minorticks_on()
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

    print('Mean Absolute Error:', round(MAE, 2))
    # print('Max Error:', round(MAX, 2))
    # print('Exaplined Variance:', round(EVS, 3))
    print('R2 Scoring:', round(R2, 3))
    print('RSE:', round(RSE, 3))
    print('Relative:', np.round(np.mean(REL), 2), '%.')
    print('\nAccuracy:', np.round(100-np.mean(REL), 2), '%.')