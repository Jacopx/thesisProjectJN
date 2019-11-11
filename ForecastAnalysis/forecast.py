import glob
import time
import pandas as pd
import datetime
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")


def random_forest(dbc, file):
    test_size = 0.25
    predictor = 1000
    random = 12

    features = pd.read_csv(file + '.csv', parse_dates=True)

    print('The shape of our features is:', features.shape)

    # Descriptive statistics for each column
    features['date'] = pd.to_datetime(features['date'], format="%Y-%m-%d")
    features['wday'] = features.date.dt.weekday
    features['day'] = features.date.dt.day
    features['month'] = features.date.dt.month
    features['year'] = features.date.dt.year
    features = features.drop('date', axis=1)

    labels = np.array(features['n'])
    mean = np.mean(labels)
    features = features.drop('n', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=test_size, random_state=random, shuffle=False)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    rf = RandomForestRegressor(n_estimators=predictor, random_state=random, verbose=1, n_jobs=-1)
    rf.fit(train_features, train_labels)

    # The baseline predictions are the historical averages
    baseline_errors = abs(mean - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    predictions = rf.predict(test_features)

    predictions = np.round(predictions, decimals=0)

    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    rel = round(np.mean(errors), 2) / np.mean(test_labels)
    print('Relative:', round(rel, 2) * 100, '%.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)  # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    rse = mean_squared_error(test_labels, predictions)
    print('RSE:', round(rse, 3))

    # Get numerical feature importances
    importances = list(rf.feature_importances_)  # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 4)) for feature, importance in
                           zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    with open(file[5:] + '-' + str(test_size) + '.txt', "w") as f:
        print('ESTIMATOR: ' + str(predictor), file=f)
        print('RANDOM: ' + str(random), file=f)
        print('AVGERR: ' + str(round(np.mean(baseline_errors), 2)), file=f)
        print('ACC: ' + str(round(accuracy, 2)) + '%', file=f)
        print('Relative: ' + str(round(rel * 100, 2)) + '%', file=f)
        print('MAE: ' + str(round(np.mean(errors), 2)), file=f)
        print('RSE: ' + str(round(rse, 3)), file=f)
        [print('Variable: {:20} Importance: {}'.format(*pair), file=f) for pair in feature_importances]

    months = features[:, feature_list.index('month')]
    days = features[:, feature_list.index('day')]
    years = features[:, feature_list.index('year')]

    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]

    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

    true_data = pd.DataFrame(data={'date': dates, 'n': labels})
    months = test_features[:, feature_list.index('month')]
    days = test_features[:, feature_list.index('day')]
    years = test_features[:, feature_list.index('year')]

    test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
                  zip(years, months, days)]

    test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in
                  test_dates]

    predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})

    true_data.drop_duplicates(subset='date', keep='first', inplace=True)
    t = pd.merge(true_data, predictions_data, on='date', how='left')
    r = pd.merge(true_data, predictions_data, on='date', how='right')
    print('DF Merged: ' + str(len(t)))

    plt.figure(figsize=(40, 25))
    sns.lineplot(true_data['date'], true_data['n'], label='real')
    sns.lineplot(t['date'], t['prediction'], label='predict')
    sns.lineplot(t['date'], mean, label='mean')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Date')
    plt.ylabel('Event')
    plt.title(file[5:] + '-' + str(test_size) + '-all')
    plt.savefig(file[5:] + '-' + str(test_size) + '-all.png', dpi=300)
    plt.show()

    plt.figure(figsize=(40, 25))
    sns.lineplot(r['date'], r['n'], label='real')
    sns.lineplot(r['date'], r['prediction'], label='predict')
    sns.lineplot(r['date'], mean, label='mean')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Date')
    plt.ylabel('Event')
    plt.title(file[5:] + '-' + str(test_size) + '-specific')
    plt.savefig(file[5:] + '-' + str(test_size) + '-specific.png', dpi=300)
    plt.show()
