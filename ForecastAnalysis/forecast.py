import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def random_forest(dbc, csv_name):
    features = pd.read_csv(csv_name, parse_dates=True)

    features['daily'] = features['n']

    temp = pd.pivot_table(features, values=['daily'], index=['date'], aggfunc='sum')

    print('The shape of our features is:', features.shape)

    features = features.merge(temp, left_on=['date'], right_on=['date'])

    # Descriptive statistics for each column
    features['date'] = pd.to_datetime(features['date'], format="%Y-%m-%d")
    features['wday'] = features.date.dt.weekday
    features['day'] = features.date.dt.day
    features['month'] = features.date.dt.month
    features['year'] = features.date.dt.year
    features = features.drop('date', axis=1)

    labels = np.array(features['n'])
    features = features.drop('n', axis=1)  # Saving feature names for later use
    features = features.drop('daily_x', axis=1)  # Saving feature names for later use
    features = features.drop('daily_y', axis=1)  # Saving feature names for later use
    feature_list = list(features.columns)  # Convert to numpy array
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.25, random_state=43, shuffle=False)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # # The baseline predictions are the historical averages
    # baseline_preds = test_features[:, feature_list.index('daily_y')]
    # baseline_errors = abs(baseline_preds - test_labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=500, random_state=43, verbose=1, n_jobs=-1)  # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)  # Calculate the absolute errors
    errors = abs(predictions - test_labels)  # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    rel = np.mean(errors) / np.mean(test_labels)
    print('RE:', round(rel * 100, 2), '%.')

    rse = mean_squared_error(test_labels, predictions)
    print('RSE:', rse)

    # Get numerical feature importances
    importances = list(rf.feature_importances_)  # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in
                           zip(feature_list, importances)]  # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)  # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # Use datetime for creating date objects for plotting
    months = features[:, feature_list.index('month')]
    days = features[:, feature_list.index('day')]
    years = features[:, feature_list.index('year')]  # List and then convert to datetime object
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]  # Dataframe with true values and dates
    true_data = pd.DataFrame(data={'date': dates, 'n': labels})  # Dates of predictions
    months = test_features[:, feature_list.index('month')]
    days = test_features[:, feature_list.index('day')]
    years = test_features[:, feature_list.index('year')]  # Column of dates
    test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
                  zip(years, months, days)]  # Convert to datetime objects
    test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in
                  test_dates]  # Dataframe with predictions and dates
    predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})  # Plot the actual values
    plt.figure(figsize=(30, 15))

    t = predictions_data.merge(true_data, left_on=['date'], right_on=['date'])
    plt.plot(true_data['date'], true_data['n'], 'b-', label='real')  # Plot the predicted values
    plt.plot(t['date'], t['prediction'], 'ro', label='predict')
    plt.xticks(rotation='60')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Event')
    plt.title('Actual and Predicted Values')

    plt.show()