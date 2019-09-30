import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from datetime import datetime


def replace_camelcase_with_underscore(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)


def clean_col_names(df):
    df.columns = [name.lstrip() for name in df.columns.tolist()]
    df.columns = [name.rstrip() for name in df.columns.tolist()]
    df.columns = [name.replace(' ', '') for name in df.columns.tolist()]
    df.columns = [name.replace('_', '') for name in df.columns.tolist()]
    df.columns = [replace_camelcase_with_underscore(name) for name in df.columns.tolist()]
    df.columns = [name.lower() for name in df.columns.tolist()]
    return df


def read_data():
    print('Data read...', end='')
    path = 'data'  # use your path
    # all_files = glob.glob(path + "/operationsSFFD_RAW.csv")
    all_files = glob.glob(path + "/operationsSFFD.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, low_memory=False, parse_dates=True, error_bad_lines=False)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    print(' OK')
    return frame


def data_reduction(df):
    return df[['unit_id', 'call_type', 'call_type_group', 'received_dt_tm', 'available_dt_tm', 'zipcodeof_incident', 'battalion', 'station_area', 'box', 'final_priority', 'location']]


def parser(df):
    print('Parser', end='')
    df['start_dt'] = pd.to_datetime(df['received_dt_tm'], format="%m/%d/%Y %H:%M:%S %p")
    print('...', end='')
    df['end_dt'] = pd.to_datetime(df['available_dt_tm'], format="%m/%d/%Y %H:%M:%S %p")
    print(' OK')


def feature_extraction(df):
    print('Feature extraction START', end='')
    # Extract date, month, hour of start
    df['start_day'] = df['start_dt'].dt.day
    print('.', end='')
    df['start_month'] = df['start_dt'].dt.month
    print('.', end='')
    df['start_hour'] = df['start_dt'].dt.hour
    print('.', end='')
    df['start_day_of_week'] = df['start_dt'].dt.weekday
    print(' OK')

    print('Feature extraction END', end='')
    # Extract date, month, hour of end
    df['end_day'] = df['end_dt'].dt.day
    print('.', end='')
    df['end_month'] = df['end_dt'].dt.month
    print('.', end='')
    df['end_hour'] = df['end_dt'].dt.hour
    print('.', end='')
    df['end_day_of_week'] = df['end_dt'].dt.weekday
    print(' OK')

    print('Feature extraction DURATION', end='')
    # Extract date, month, hour of end
    df['duration'] = df['end_dt'] - df['start_dt']
    print('... OK')


def remove_nan(df):
    df = df[['unit_id', 'call_type', 'call_type_group', 'start_dt', 'end_dt', 'zipcodeof_incident', 'battalion', 'station_area', 'box', 'final_priority', 'location']]
    print("End NaN rows: " + str((df['end_dt'].isnull().sum() / df.size) * 100) + '%')
    print('Removing Nan rows...', end='')
    df = df[np.isfinite(df['end_dt'])]
    print(' OK')
    print("End NaN rows: " + str((df['end_dt'].isnull().sum() / df.size) * 100) + '%')
    return df


def main():
    t0 = time.time()
    print("Starting Analysis...\n")

    df = read_data()
    # clean_col_names(df)
    # df = data_reduction(df)
    # df.to_csv('data/operations.csv', index=False)

    parser(df)
    df = remove_nan(df)
    feature_extraction(df)

    print("Total Time [{} s]".format(round(time.time() - t0, 2)))
    # print("Number of Rows [{}]".format(tot_row))


if __name__ == "__main__":
    main()
