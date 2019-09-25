import time
import numpy
import pandas as pd
import math
import scipy

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


def main():
    data_op = 'data/operations.csv'
    df = pd.read_csv(data_op, parse_dates=True, error_bad_lines=False)

    print(df)

    dict_typology(df)
    dict_loc(df)

    df = extract_feature(df)

    print(df.head(15))


if __name__ == "__main__":
    main()
