import pandas as pd
import datetime
import matplotlib
matplotlib.use('Agg')
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
    predictor = 240
    random = 12

    time_horizons = [5, 15, 30, 45, 60, 75, 90, 105, 120, 180, 360]
    # time_horizons = [5, 15]
    # time_horizons = [5]

    maes = []
    rels = []
    accs = []
    rses = []

    # for horizon in time_horizon:

    features_basic = pd.read_csv(file + '.csv', parse_dates=True, index_col=3)

    print('################################################')
    print('FILE:', file, '\n')
    print('The shape of our features is:', features_basic.shape)

    for horizon in time_horizons:
        print('\nTIME HORIZON: {}\n'.format(horizon))
        features = features_basic.copy()
        features['n'] = features['bike_available'].shift(-horizon, fill_value=-1)
        features = features.head(-horizon)

        # Descriptive statistics for each column
        features.index = pd.to_datetime(features.index, format="%Y-%m-%d %H:%M:%S")
        features['wday'] = features.index.dayofweek
        features['day'] = features.index.day
        features['month'] = features.index.month
        features['year'] = features.index.year
        features['m'] = features.index.minute
        features['h'] = features.index.hour

        features['time'] = features['m'] + features['h'] * 60
        features = features.drop('station_id', axis=1)
        features = features.drop('docks_available', axis=1)

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
        test_labels = test_labels[0:len(predictions)]

        predictions = np.round(predictions, decimals=0)

        errors = abs(predictions - test_labels)
        print('Mean Absolute Error:', round(np.mean(errors), 2))
        maes.append(round(np.mean(errors), 2))

        rel = round(np.mean(errors), 2) / np.mean(test_labels)
        print('Relative:', round(rel * 100, 2), '%.')
        rels.append(round(rel * 100, 2))

        # Calculate mean absolute percentage error (MAPE)
        # mape = 100 * (errors / test_labels)  # Calculate and display accuracy
        mape = 100 * (errors / test_labels)
        accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')
        accs.append(round(accuracy, 2))

        rse = mean_squared_error(test_labels, predictions)
        print('RSE:', round(rse, 3))
        rses.append(round(rse, 3))

        # Get numerical feature importances
        importances = list(rf.feature_importances_)  # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 4)) for feature, importance in
                               zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                     reverse=True)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

        with open(file[5:] + '-' + str(horizon) + '_' + str(predictor) + '.txt', "w") as f:
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
        hours = features[:, feature_list.index('h')]
        minutes = features[:, feature_list.index('m')]

        dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) + ' ' + str(int(hour)) + ':' + str(int(minute)) for year, month, day, hour, minute in
                 zip(years, months, days, hours, minutes)]

        dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M') for date in dates]

        true_data = pd.DataFrame(data={'date': dates, 'n': labels})
        months = test_features[:, feature_list.index('month')]
        days = test_features[:, feature_list.index('day')]
        years = test_features[:, feature_list.index('year')]
        hours = test_features[:, feature_list.index('h')]
        minutes = test_features[:, feature_list.index('m')]

        test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) + ' ' + str(int(hour)) + ':' + str(int(minute)) for year, month, day, hour, minute in
                 zip(years, months, days, hours, minutes)]

        test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M') for date in
                      test_dates]

        predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})

        true_data = true_data.groupby(true_data.date.dt.date)['n'].sum()
        predictions_data = predictions_data.groupby(predictions_data.date.dt.date)['prediction'].sum()

        true_data = true_data.reset_index()
        predictions_data = predictions_data.reset_index()

        t = pd.merge(true_data, predictions_data, on='date', how='left')
        r = pd.merge(true_data, predictions_data, on='date', how='right')

        print('DF Merged: ' + str(len(t)))

        plt.figure(figsize=(35, 16))
        sns.lineplot(true_data.date, true_data['n'], label='real', ci=None)
        sns.lineplot(t.date, t['prediction'], label='predict', ci=None)
        # sns.lineplot(t.date, mean, label='mean', ci=None)
        plt.xticks(rotation='60')
        plt.legend()  # Graph labels
        plt.xlabel('Date')
        plt.ylabel('Event')
        plt.minorticks_on()
        plt.title(file[5:] + '-' + str(horizon) + '_' + str(predictor) + '-all')
        plt.savefig(file[5:] + '-' + str(horizon) + '_' + str(predictor) + '-all.png', dpi=240)
        plt.show()
        print('Plot ALL')

        plt.figure(figsize=(35, 16))
        sns.lineplot(r.date, r['n'], label='real', ci=None)
        sns.lineplot(r.date, r['prediction'], label='predict', ci=None)
        # sns.lineplot(r.date, mean, label='mean', ci=None)
        plt.xticks(rotation='60')
        plt.legend()  # Graph labels
        plt.xlabel('Date')
        plt.ylabel('Event')
        plt.minorticks_on()
        plt.title(file[5:] + '-' + str(horizon) + '_' + str(predictor) + '-specific')
        plt.savefig(file[5:] + '-' + str(horizon) + '_' + str(predictor) + '-specific.png', dpi=240)
        plt.show()
        print('Plot SPECIFIC\n')

    plt.figure(figsize=(10, 10))
    sns.lineplot(time_horizons, rses, label='RSE')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Time Horizon')
    plt.ylabel('RSE error')
    # plt.grid(axis='both', which='both')
    plt.minorticks_on()
    plt.title(file[5:] + '-' + str(predictor) + '-rse_error')
    plt.savefig(file[5:] + '-' + str(predictor) + '-rse_error.png', dpi=240)
    plt.show()
    print('\nPlot RSE ERRORS')

    plt.figure(figsize=(10, 10))
    sns.lineplot(time_horizons, maes, label='MAE')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Time Horizon')
    plt.ylabel('MAE error')
    # plt.grid(axis='both', which='both')
    plt.minorticks_on()
    plt.title(file[5:] + '-' + str(predictor) + '-mae_error')
    plt.savefig(file[5:] + '-' + str(predictor) + '-mae_error.png', dpi=240)
    plt.show()
    print('\nPlot MAE ERRORS')

    plt.figure(figsize=(10, 10))
    sns.lineplot(time_horizons, rels, label='REL')
    # sns.lineplot(time_horizons, accs, label='ACC')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Time Horizon')
    plt.ylabel('RELATIVE error')
    # plt.grid(axis='both', which='both')
    plt.minorticks_on()
    plt.title(file[5:] + '-' + str(predictor) + '-rel_error')
    plt.savefig(file[5:] + '-' + str(predictor) + '-rel_error.png', dpi=240)
    plt.show()
    print('\nPlot RELATIVE ERRORS')

    with open(file[5:] + '-' + str(predictor) + '-errors.txt', 'w') as f:
        print(maes, file=f)
        print(rels, file=f)
        print(accs, file=f)
        print(rses, file=f)
