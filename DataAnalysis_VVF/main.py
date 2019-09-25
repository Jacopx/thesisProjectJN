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
    df['month'] = df["date"].dt.month
    df['day'] = df["date"].dt.day

    cols = df.columns.tolist()
    cols = cols[:2] + cols[-2:-1] + [cols[-1]] + cols[2:-2]
    df = df[cols]
    return df


def convert_time(df):
    v = df['start'].str.split(':', expand=True).astype(int)
    s = pd.to_timedelta(v[0], unit='h') + pd.to_timedelta(v[1], unit='s')
    df['start'] = s.astype(int)

    v = df['finish'].str.split(':', expand=True).astype(int)
    s = pd.to_timedelta(v[0], unit='h') + pd.to_timedelta(v[1], unit='s')
    df['finish'] = s.astype(int)


def convert_gps(df):
    df.ffill(axis=1, inplace=True)
    df['X'] = df["x"].astype(float)
    df['Y'] = df["y"].astype(float)
    print(df)


def main():
    data_op = 'data/operations.csv'
    df = pd.read_csv(data_op, parse_dates=True, error_bad_lines=False)

    dict_typology(df)
    dict_loc(df)
    convert_time(df)
    # convert_gps(df)

    df = extract_feature(df)

    print(df.dtypes)
    print(df)
    corr = df.corr()

    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, cmap="YlOrRd", annot=True, linewidths=0.3, vmin=-1, square=False)
    plt.show()


if __name__ == "__main__":
    main()
