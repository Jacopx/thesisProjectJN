import time
import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

# Defining a color pattern based on Tableau's color code
colrcode = [(31, 119, 180), (255, 127, 14),
             (44, 160, 44), (214, 39, 40),
             (148, 103, 189),  (140, 86, 75),
             (227, 119, 194), (127, 127, 127),
             (188, 189, 34), (23, 190, 207)]


def replace_camelcase_with_underscore(name):
    '''
    Taken from Stack Overflow:
    http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)


def clean_col_names(df):
    df.columns = [name.lstrip() for name in df.columns.tolist()]
    df.columns = [name.rstrip() for name in df.columns.tolist()]
    df.columns = [name.replace(' ','') for name in df.columns.tolist()]
    df.columns = [name.replace('_','') for name in df.columns.tolist()]
    df.columns = [replace_camelcase_with_underscore(name) for name in df.columns.tolist()]
    return df


def main():
    print('Starting analysis...')
    t0 = time.time()
    print('Data read...', end='')
    # Read data from csv files
    # weath_df = pd.read_csv("data/weather.csv", nrows=None)
    trip_df = pd.read_csv("data/trip.csv", nrows=None)
    stn_df = pd.read_csv("data/station.csv", nrows=None)
    # stat_df = pd.read_csv("data/status.csv", nrows=None)
    print(' OK')


    # Remove outliers. Outliers defined as values greater than 99.5th percentile
    maxVal = np.percentile(trip_df['duration'], 99.5)
    trip_df = trip_df[trip_df['duration'] <= maxVal]

    print('Feature extraction', end='')
    # Extract date, month, hour of start
    trip_df['start_day'] = trip_df['start_date'].map(lambda x: (datetime.strptime(x, "%m/%d/%Y %H:%M")).day)
    print('.', end='')
    trip_df['start_month'] = trip_df['start_date'].map(lambda x: (datetime.strptime(x, "%m/%d/%Y %H:%M")).month)
    print('.', end='')
    trip_df['start_hour'] = trip_df['start_date'].map(lambda x: (datetime.strptime(x, "%m/%d/%Y %H:%M")).hour)
    print('.', end='')
    trip_df['day_of_week'] = trip_df['start_date'].map(lambda x: (datetime.strptime(x, "%m/%d/%Y %H:%M")).weekday())
    print(' OK')

    # Get only a subset of the data
    trip_df = trip_df[['id', 'start_day', 'start_month', 'start_hour', 'day_of_week', 'duration', 'start_station_id', 'end_station_id', 'subscription_type', 'zip_code']]
    stn_df = stn_df[['station_id', 'lat', 'long', 'dock_count']]

    fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)

    print('Group by Subscriber... ', end='')
    a = trip_df[trip_df['subscription_type'] == 'Subscriber'].groupby(['day_of_week', 'start_hour'])['id'].count()
    print(' OK')
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    plt.sca(axes[0])

    print('Plot Subscriber... ', end='')
    for i in range(7):
        sns.lineplot(data=a[i], label=days[i])

    plt.title('Number of trips by hour : Subscriber')
    plt.legend()
    print(' OK')

    plt.minorticks_on()
    plt.tight_layout()
    plt.grid()

    print('Group by Customer... ', end='')
    a = trip_df[trip_df['subscription_type'] == 'Customer'].groupby(['day_of_week', 'start_hour'])['id'].count()
    print(' OK')

    print('Plot Customer... ', end='')
    plt.sca(axes[1])

    for i in range(7):
        sns.lineplot(data=a[i], label=days[i])

    plt.title('Number of trips by hour : Customer')
    plt.legend()
    print(' OK')

    plt.minorticks_on()
    plt.grid()

    plt.savefig('plot/num_trips.png', dpi=300)
    plt.show()

    plt.figure(figsize=[8, 6])
    plt.axvline(x=1800, color='r', linestyle='--')

    print('Plot Subscriber duration... ', end='')
    sub = trip_df[(trip_df['subscription_type'] == 'Subscriber') & (trip_df['duration'] < 2800)]
    sns.distplot(sub['duration'], label='Subscriber', kde=False)
    print(' OK')

    print('Plot Customer duration... ', end='')
    cos = trip_df[(trip_df['subscription_type'] == 'Customer') & (trip_df['duration'] < 2800)]
    sns.distplot(cos['duration'], label='Casual', kde=False)
    print(' OK')

    plt.title('Histogram: Duration in sec')
    plt.xlabel('Duration of ride (s)')
    plt.legend()
    plt.minorticks_on()
    plt.grid()

    plt.savefig('plot/dur_trips.png', dpi=300)
    plt.show()

    print('Making calculation... ', end='')
    cosC = trip_df[(trip_df['subscription_type'] == 'Customer') & (trip_df['duration'] > 1800)]
    val_cas_over = cosC['duration'].count() / float(trip_df[(trip_df['subscription_type'] == 'Customer')]['duration'].count())

    subC = trip_df[(trip_df['subscription_type'] == 'Subscriber') & (trip_df['duration'] > 1800)]
    val_sub_over = subC['duration'].count() / float(trip_df[(trip_df['subscription_type'] == 'Subscriber')]['duration'].count())
    print(' OK')

    sub = trip_df[(trip_df['subscription_type'] == 'Subscriber')]

    print('Plot Subscriber matrix usage... ', end='')
    counts_end_station = sub.groupby(['start_station_id', 'end_station_id'])['id'].count()
    counts_end_station = counts_end_station.unstack()
    plt.figure(figsize=(12, 10))
    sns.heatmap(counts_end_station, cmap='YlOrRd', linewidths=0, square=False, vmax=450)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.title('Traffic between pairs of bike stations: Subscribers')
    plt.savefig('plot/stat_usage_sub.png', dpi=300)
    plt.show()
    print(' OK')

    cos = trip_df[(trip_df['subscription_type'] == 'Customer')]

    print('Plot Customer matrix usage... ', end='')
    counts_end_station = cos.groupby(['start_station_id', 'end_station_id'])['id'].count()
    counts_end_station = counts_end_station.unstack()
    plt.figure(figsize=(12, 10))
    sns.heatmap(counts_end_station, cmap='YlGnBu', linewidths=0, square=True, vmax=450)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.title('Traffic between pairs of bike stations: Customers')
    plt.savefig('plot/stat_usage_cos.png', dpi=300)
    plt.show()
    print(' OK')

    print('Analysis terminated: ' + str(time.time() - t0) + 's\n')

    print("{0}% of causal customers pay overtime fee! ".format(np.round(val_cas_over * 100, 0)))
    print("{0}% of subscribers pay overtime fee! ".format(np.round(val_sub_over * 100, 0)))


if __name__ == "__main__":
    main()
