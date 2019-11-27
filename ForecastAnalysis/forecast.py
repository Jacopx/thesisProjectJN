import pandas as pd
import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import mean_squared_error
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")

test_size = 0.25
predictor = 400
random = 12
n_jobs = 6

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

    plot(file, test_labels, predictions)
    importances(model, feature_list)
    errors(test_labels, predictions, mean)


def count_model(file):

    features_basic = pd.read_csv(file + '.csv')
    features_basic = features_basic.drop('index', axis=1)  # Saving feature names for later use

    infos(file, features_basic)

    features = features_basic.copy()
    features = features.dropna()

    features['n'] = features['n'].astype('int32')

    labels = np.array(features['n'])
    mean = np.mean(labels)
    features = features.drop('n', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=test_size, random_state=random, shuffle=False)

    ######################### MODEL DEFINITIONS ############################

    model = RandomForestRegressor(n_estimators=predictor, random_state=random, verbose=0, n_jobs=n_jobs)
    # model = GradientBoostingRegressor(n_estimators=predictor, random_state=random, verbose=0)
    # model = MLPRegressor(verbose=1)
    model.fit(train_features, train_labels)

    ######################### MODEL DEFINITIONS ############################

    predictions = model.predict(test_features)

    plot(file, test_labels, predictions)
    importances(model, feature_list)
    errors(test_labels, predictions, mean)


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

    plt.figure(figsize=(100, 25))
    sns.lineplot(n, test_labels, label='real')
    sns.lineplot(n, predictions, label='predict')
    # sns.lineplot(r.date, mean, label='mean', ci=None)
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Issue')
    plt.ylabel('n')
    plt.minorticks_on()
    plt.title(file + ' predictions')
    plt.savefig(file + '_predictions.png', dpi=240)
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
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    test_labels = test_labels[0:len(predictions)]

    predictions = np.round(predictions, decimals=0)

    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2))

    rel = round(np.mean(errors), 2) / np.mean(test_labels)
    print('Relative:', round(rel * 100, 2), '%.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    rse = mean_squared_error(test_labels, predictions)
    print('RSE:', round(rse, 3))