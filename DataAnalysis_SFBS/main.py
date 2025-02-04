import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def remove_outliers(df):
    # Remove outliers. Outliers defined as values greater than 99.5th percentile
    maxVal = np.percentile(df['duration'], 99.5)
    df = df[df['duration'] <= maxVal]


def feature_extraction(df):
    print('Feature extraction', end='')
    # Extract date, month, hour of start
    df['start_dt'] = pd.to_datetime(df['start_date'], format="%m/%d/%Y %H:%M")
    df['end_dt'] = pd.to_datetime(df['end_date'], format="%m/%d/%Y %H:%M")

    df['start_day'] = df['start_dt'].dt.day
    print('.', end='')
    df['start_month'] = df['start_dt'].dt.month
    print('.', end='')
    df['start_hour'] = df['start_dt'].dt.hour
    print('.', end='')
    df['day_of_week'] = df['start_dt'].dt.weekday
    df['week'] = df['start_dt'].dt.week
    df['year'] = df['start_dt'].dt.year
    print(' OK')


def reduce_set(trip_df, stn_df):
    # Get only a subset of the data
    trip_df = trip_df[['id', 'start_day', 'start_month', 'start_hour', 'day_of_week', 'duration', 'start_station_id',
                       'end_station_id', 'subscription_type', 'zip_code']]
    stn_df = stn_df[['station_id', 'lat', 'long', 'dock_count']]


def line_plot(df):
    fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)

    print('Group by Subscriber... ', end='')
    a = df[df['subscription_type'] == 'Subscriber'].groupby(['day_of_week', 'start_hour'])['id'].count()
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
    a = df[df['subscription_type'] == 'Customer'].groupby(['day_of_week', 'start_hour'])['id'].count()
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


def duration_plot(df):
    plt.figure(figsize=[8, 6])
    plt.axvline(x=1800, color='r', linestyle='--')

    print('Plot Subscriber duration... ', end='')
    sub = df[(df['subscription_type'] == 'Subscriber') & (df['duration'] < 2800)]
    sns.distplot(sub['duration'], label='Subscriber', kde=False)
    print(' OK')

    print('Plot Customer duration... ', end='')
    cos = df[(df['subscription_type'] == 'Customer') & (df['duration'] < 2800)]
    sns.distplot(cos['duration'], label='Casual', kde=False)
    print(' OK')

    plt.title('Histogram: Duration in sec')
    plt.xlabel('Duration of ride (s)')
    plt.legend()
    plt.minorticks_on()
    plt.grid()

    plt.savefig('plot/dur_trips.png', dpi=300)
    plt.show()


def calculation(df):
    print('Making calculation... ', end='')
    cosC = df[(df['subscription_type'] == 'Customer') & (df['duration'] > 1800)]
    val_cas_over = cosC['duration'].count() / float(df[(df['subscription_type'] == 'Customer')]['duration'].count())

    subC = df[(df['subscription_type'] == 'Subscriber') & (df['duration'] > 1800)]
    val_sub_over = subC['duration'].count() / float(df[(df['subscription_type'] == 'Subscriber')]['duration'].count())
    print(' OK')


def matrix_start_end(df, type):
    sub = df[(df['subscription_type'] == type)]

    print('Plot ' + type + 'matrix usage... ', end='')
    counts_end_station = sub.groupby(['start_station_id', 'end_station_id'])['id'].count()
    counts_end_station = counts_end_station.unstack()
    plt.figure(figsize=(12, 10))
    sns.heatmap(counts_end_station, cmap='YlOrRd', linewidths=0, square=False, vmax=450)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.title('Traffic between pairs of bike stations: ' + type)
    plt.savefig('plot/stat_usage_' + type + '.png', dpi=300)
    plt.show()
    print(' OK')


def correlation_map(df):
    df.loc[df['subscription_type'] == 'Customer', 'subscription_type'] = 0
    df.loc[df['subscription_type'] == 'Subscriber', 'subscription_type'] = 1

    df['subscription_type'] = df['subscription_type'].astype(int)

    print('Plot Correlation Matrix... ', end='')
    plt.figure(figsize=(12, 12))
    corr = df.corr()
    sns.heatmap(corr, cmap='seismic', annot=True, linewidths=0.2, vmin=-1, vmax=1, square=True)
    plt.title('Correlation Map')
    plt.ylabel('Features')
    plt.xlabel('Features')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/correlation_map.png', dpi=300)
    plt.show()
    print(' OK')


def year_month(df):
    group = df.groupby(['year', 'start_month', 'start_hour'])['id'].count()
    group.to_csv('play.csv', index=True)


def merge(trip_df, stn_df):
    m1 = trip_df.merge(stn_df, left_on='start_station_id', right_on='station_id')
    m1r = m1.rename(columns={"dock_count": 'start_dock_count', "lat": "start_lat", 'long': 'start_long'})
    m2 = m1r.merge(stn_df, left_on='end_station_id', right_on='station_id')
    m2r = m2.rename(columns={"dock_count": 'end_dock_count', "lat": "end_lat", 'long': 'end_long'})
    m = m2r[['id', 'duration',
             'start_dt', 'start_station_name', 'start_station_id', 'start_dock_count', 'start_lat', 'start_long',
             'end_dt', 'end_station_name', 'end_station_id', 'end_dock_count', 'end_lat', 'end_long',
             'bike_id', 'subscription_type', 'start_day', 'start_month', 'start_hour', 'day_of_week', 'week']]

    m.to_csv('data/merged_data.csv', index=False)


def main():
    print('Starting analysis...')
    t0 = time.time()

    # Read data from csv files
    print('Data read...', end='')
    trip_df = pd.read_csv("data/trip.csv", nrows=None)
    stn_df = pd.read_csv("data/station.csv", nrows=None)
    print(' OK')

    remove_outliers(trip_df)
    feature_extraction(trip_df)
    reduce_set(trip_df, stn_df)

    # merge(trip_df, stn_df)

    line_plot(trip_df)
    duration_plot(trip_df)
    calculation(trip_df)
    matrix_start_end(trip_df, 'Subscriber')
    matrix_start_end(trip_df, 'Customer')
    correlation_map(trip_df)

    year_month(trip_df)

    print('Analysis terminated: ' + str(time.time() - t0) + 's\n')


def status():
    status = pd.read_csv('data/station70.csv', nrows=None, parse_dates=True)
    status['time'] = pd.to_datetime(status['time'], format="%Y-%m-%d %H:%M:%S")
    status = status.drop('docks_available', axis=1)
    status = status.drop('station_id', axis=1)
    # status = status.rename(columns={'bikes_available': 'n'})
    status['index'] = 0

    status['mov_avg2'] = status['bikes_available'].rolling(2).mean()
    status['mov_avg2'] = status['mov_avg2'].shift(1, fill_value=-1)
    status = status.tail(-1)

    # status['mov_avg3'] = status['bikes_available'].rolling(3).mean()
    # status['mov_avg3'] = status['mov_avg3'].shift(1, fill_value=-1)
    # status = status.tail(-2)

    status['mov_avg5'] = status['bikes_available'].rolling(5).mean()
    status['mov_avg5'] = status['mov_avg5'].shift(1, fill_value=-1)
    status = status.tail(-4)

    status['mov_avg15'] = status['bikes_available'].rolling(15).mean()
    status['mov_avg15'] = status['mov_avg15'].shift(1, fill_value=-1)
    status = status.tail(-14)

    status['n'] = status['bikes_available'].shift(-15, fill_value=-1)
    status = status.head(-15)

    status['mon'] = status['time'].dt.month
    status['d'] = status['time'].dt.day
    status['wd'] = status['time'].dt.weekday
    status['h'] = status['time'].dt.hour
    status['m'] = status['time'].dt.minute
    status = status.drop('time', axis=1)

    status.to_csv('data/station70_forecast.csv', index=None)
    print('ok')


if __name__ == "__main__":
    # main()
    status()
