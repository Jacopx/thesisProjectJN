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


def convert_time(df):
    v = df['start'].str.split(':', expand=True).astype(int)
    s = pd.to_timedelta(v[0], unit='h') + pd.to_timedelta(v[1], unit='s')
    df['start'] = s.astype(int)

    v = df['finish'].str.split(':', expand=True).astype(int)
    s = pd.to_timedelta(v[0], unit='h') + pd.to_timedelta(v[1], unit='s')
    df['finish'] = s.astype(int)


def convert_gps(df):
    df['x'] = df["x"].astype(float)
    df['y'] = df["y"].astype(float)


def correlation_map(df):
    df_corr = df[['day_year', 'month', 'day', 'start', 'finish', 'duration', 'x', 'y', 'locat', 'typo']]
    corr = df_corr.corr()

    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, cmap="YlOrRd", annot=True, linewidths=0.2, vmin=-1, square=False)

    plt.title('Correlation Map')
    plt.ylabel('Features')
    plt.xlabel('Features')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('correlation_map.png', dpi=300)
    plt.show()


def main():
    data_op = 'data/operations.csv'
    df = pd.read_csv(data_op, parse_dates=True, error_bad_lines=False)

    dict_typology(df)
    dict_loc(df)
    convert_time(df)
    convert_gps(df)
    extract_feature(df)

    correlation_map(df)


if __name__ == "__main__":
    main()
