import time
import numpy as np
import pandas as pd
import math
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

loc = {}
typo = {}


def dict_typology(df):
    typo_dict = dict(df.typo)

    i = 0
    for t in typo_dict.values():
        if t not in typo.keys():
            typo[t] = i
            i = i + 1

    df["typo"].replace(typo, inplace=True)


def dict_loc(df):
    loc_dict = dict(df.locat)

    i = 0
    for t in loc_dict.values():
        if t not in loc.keys():
            loc[t] = i
            i = i + 1

    df["locat"].replace(loc, inplace=True)


def extract_feature(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day_year'] = df["date"].dt.dayofyear
    df['month'] = df["date"].dt.month
    df['day'] = df["date"].dt.day
    df['week'] = df["date"].dt.week
    df['year'] = df["date"].dt.year

# @TODO: Need to fix the duration with the new available date
def convert_time(df):
    v = df['start'].str.split(':', expand=True).astype(int)
    s = pd.to_timedelta(v[0], unit='h') + pd.to_timedelta(v[1], unit='s')
    df['start'] = s.astype(int)

    v = df['finish'].str.split(':', expand=True).astype(int)
    s = pd.to_timedelta(v[0], unit='h') + pd.to_timedelta(v[1], unit='s')
    df['finish'] = s.astype(int)

    df['dur'] = (df['finish'] - df['start']) / 1000000000
    df['dur'] = df['dur'].astype(int)

    print(df[['duration', 'dur']])


def convert_gps(df):
    df['x'] = df["x"].astype(float)
    df['y'] = df["y"].astype(float)


def correlation_map(df_input):
    # Computing number of operations per week
    # week_count = df_input[['week', 'n']].groupby(['week']).count()

    # Computing number of operations per year per week
    week_count = df_input[['week', 'n', 'year']].groupby(['year', 'week']).count()

    # Restrict dataframe
    complete = df_input[['year', 'week', 'day_year', 'month', 'day', 'start', 'finish', 'dur', 'duration', 'x', 'y', 'locat', 'typo']]

    # Make merge
    # df = pd.merge(complete, week_count, on=['week'])
    df = pd.merge(complete, week_count, on=['year', 'week'])

    corr = df.corr()

    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, cmap='seismic', annot=True, linewidths=0.2, vmin=-1, vmax=1, square=False)

    plt.title('Correlation Map')
    plt.ylabel('Features')
    plt.xlabel('Features')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/correlation_map.png', dpi=300)
    plt.show()


def distribution(df_input):
    df = df_input[['day_year', 'month', 'day', 'start', 'finish', 'duration', 'x', 'y', 'locat', 'typo']]
    bins = [3, 4, 5, 8, 10, 18]
    color = ['b', 'r', 'gray', 'g', 'orange', 'purple']

    for (columnName, columnData) in df.iteritems():
        i = 1
        plt.figure(figsize=(12, 8))
        for batch in bins:
            plt.subplot(2, 3, i)
            sns.distplot(columnData, bins=batch, color=color[(i-1) % 6])
            plt.minorticks_on()
            plt.tight_layout()
            plt.grid()
            plt.title('Distribution ' + columnName + ' [' + str(batch) + ']')
            i = i + 1

        plt.savefig('plot/distribution_' + columnName + '.png', dpi=300)
        # plt.show()


def main():
    t0 = time.time()
    print("Starting Analysis...\n")
    data_op = 'data/operations.csv'
    df = pd.read_csv(data_op, parse_dates=True, error_bad_lines=False)

    dict_typology(df)
    dict_loc(df)
    convert_time(df)
    convert_gps(df)
    extract_feature(df)
    print("Pre analysis complete [{} s]".format(round(time.time() - t0, 2)))
    print("\nCreating plots:")

    correlation_map(df)
    print("\t * Correlation Matrix [{} s]".format(round(time.time() - t0, 2)))
    # distribution(df)
    # print("\t * Distribution [{} s]".format(round(time.time() - t0, 2)))


if __name__ == "__main__":
    main()
