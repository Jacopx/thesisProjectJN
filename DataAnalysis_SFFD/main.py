import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from datetime import datetime

unit = {}
loc = {}
subtypo = {'Potentially Life-Threatening': 1, 'Non Life-threatening': 2, 'Alarm': 3, 'Fire': 4}
typo = {'Medical Incident': 0, 'Outside Fire': 1, 'Alarms': 2, 'Citizen Assist / Service Call': 3, 'Traffic Collision': 4,
        'Other': 5, 'Structure Fire': 6, 'Smoke Investigation (Outside)': 7, 'Electrical Hazard': 8, 'Elevator / Escalator Rescue': 9,
        'Vehicle Fire': 10, 'Gas Leak (Natural and LP Gases)': 11, 'Water Rescue': 12, 'Odor (Strange / Unknown)': 13, 'Fuel Spill': 14,
        'Train / Rail Incident': 15, 'Administrative': 16, 'Marine Fire': 17, 'Industrial Accidents': 18, 'High Angle Rescue': 19,
        'HazMat': 20, 'Explosion': 21, 'Confined Space / Structure Collapse': 22, 'Assist Police': 23, 'Extrication / Entrapped (Machinery, Vehicle)': 24,
        'Watercraft in Distress': 25, 'Suspicious Package': 26, 'Train / Rail Fire': 27, 'Mutual Aid / Assist Outside Agency': 28,
        'Lightning Strike (Investigation)': 29, 'Aircraft Emergency': 30, 'Oil Spill': 31}
# typo = {0: 'Medical Incident', 1: 'Outside Fire', 2: 'Alarms', 3: 'Citizen Assist / Service Call', 4: 'Traffic Collision',
#         5: 'Other', 6: 'Structure Fire', 7: 'Smoke Investigation (Outside)', 8: 'Electrical Hazard', 9: 'Elevator / Escalator Rescue',
#         10: 'Vehicle Fire', 11: 'Gas Leak (Natural and LP Gases)', 12: 'Water Rescue', 13: 'Odor (Strange / Unknown)',
#         14: 'Fuel Spill', 15: 'Train / Rail Incident', 16: 'Administrative', 17: 'Marine Fire', 18: 'Industrial Accidents',
#         19: 'High Angle Rescue', 20: 'HazMat', 21: 'Explosion', 22: 'Confined Space / Structure Collapse', 23: 'Assist Police',
#         24: 'Extrication / Entrapped (Machinery, Vehicle)', 25: 'Watercraft in Distress', 26: 'Suspicious Package', 27: 'Train / Rail Fire',
#         28: 'Mutual Aid / Assist Outside Agency', 29: 'Lightning Strike (Investigation)', 30: 'Aircraft Emergency', 31: 'Oil Spill'}


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
    # all_files = glob.glob(path + "/operationsSFFD.csv")
    all_files = glob.glob(path + "/operationsSFFD_CLEAN.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, low_memory=False, parse_dates=True, error_bad_lines=False)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    print(' OK')
    return frame


def data_reduction(df):
    return df[['unit_id', 'call_type', 'call_type_group', 'received_dt_tm', 'on_scene_dt_tm', 'available_dt_tm', 'zipcodeof_incident', 'battalion', 'station_area', 'box', 'priority', 'location']]


def remove_outliers(df, col):
    print('Remove OUTLIERS', end='')
    # Remove outliers. Outliers defined as values greater than 99.5th percentile
    maxVal = np.percentile(df[col], 98.5)
    print('.', end='')
    df = df[df[col] <= maxVal]
    print('.. OK')


def fix_priority(df_in):
    df = df_in.copy()
    df.priority.replace(['A', 'B', 'C', 'D', 'E'],  # Change 'original_priority'
                                  ['2', '2', '2', '3', '3'], inplace=True)
    df = df[(df.priority != 'I') & (df.priority != '1')]
    df.priority.dropna(axis=0, inplace=True)

    df.priority.replace(['A', 'B', 'C', 'E'],  # Change 'priority'
                         ['2', '2', '2', '3'], inplace=True)

    df.priority.astype(int)

    return df


def parser(df):
    # received_dt_tm,on_scene_dt_tm,available_dt_tm
    print('Parser', end='')
    df['rec_dt'] = pd.to_datetime(df['received_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    print('.', end='')
    df['onscene_dt'] = pd.to_datetime(df['on_scene_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    print('.', end='')
    df['end_dt'] = pd.to_datetime(df['available_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    print('. OK')


def feature_extraction(df):
    print('Feature extraction START', end='')
    # Extract date, month, hour of start
    df['rec_day'] = df['rec_dt'].dt.day
    print('.', end='')
    df['rec_month'] = df['rec_dt'].dt.month
    print('.', end='')
    df['rec_hour'] = df['rec_dt'].dt.hour
    print('.', end='')
    df['rec_day_of_week'] = df['rec_dt'].dt.weekday
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
    # Extract duration
    d = df['end_dt'] - df['rec_dt']
    print('.', end='')
    d = d / 1000000000
    print('.', end='')
    df['duration'] = d.astype(int)
    print('. OK')

    print('Feature extraction RESPONSE TIME', end='')
    # Extract duration
    d = df['onscene_dt'] - df['rec_dt']
    print('.', end='')
    d = d / 1000000000
    print('.', end='')
    df['res_time'] = d.astype(int)
    print('. OK')


def remove_nan(df):
    df = df[['unit_id', 'call_type', 'call_type_group', 'rec_dt', 'onscene_dt', 'end_dt', 'zipcodeof_incident', 'battalion', 'station_area', 'box', 'priority', 'location']]

    print("End NaN rows: " + str((df['end_dt'].isnull().sum() / df.size) * 100) + '%')
    print('Removing NaN rows...', end='')
    df = df[np.isfinite(df['end_dt'])]
    print(' OK')
    print("End NaN rows: " + str((df['end_dt'].isnull().sum() / df.size) * 100) + '%')

    print("End NaN rows: " + str((df['onscene_dt'].isnull().sum() / df.size) * 100) + '%')
    print('Removing NaN rows...', end='')
    df = df[np.isfinite(df['onscene_dt'])]
    print(' OK')
    print("End NaN rows: " + str((df['onscene_dt'].isnull().sum() / df.size) * 100) + '%')
    return df


def convert(df, dict_dest, col):
    temp_dict = dict(df[col])

    i = 0
    for t in temp_dict.values():
        if t not in dict_dest.keys():
            dict_dest[t] = i
            i = i + 1

    df[col].replace(dict_dest, inplace=True)


def corr_map(df):
    corr = df.corr()

    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, cmap='seismic', annot=True, linewidths=0.2, vmin=-1, vmax=1, square=False)

    plt.title('Correlation Map')
    plt.ylabel('Features')
    plt.xlabel('Features')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/corr_map.png', dpi=300)
    plt.show()


def distplot(df, col):
    plt.figure(figsize=(12, 12))
    sns.distplot(df[col])
    plt.show()


def main():
    t0 = time.time()
    print("Starting Analysis...\n")

    df = read_data()
    # clean_col_names(df)
    # df = data_reduction(df)
    # df.to_csv('data/operationsSFFD.csv', index=False)

    # parser(df)
    # df = remove_nan(df)
    # df = fix_priority(df)
    # feature_extraction(df)
    # remove_outliers(df, 'duration')
    # remove_outliers(df, 'res_time')
    # df.replace({"call_type": typo}, inplace=True)
    # df.replace({"call_type_group": subtypo}, inplace=True)
    # df.to_csv('data/operationsSFFD_CLEAN.csv', index=False)

    # Plots
    corr_map(df)
    distplot(df, 'duration')
    distplot(df, 'res_time')

    print("Total Time [{} s]".format(round(time.time() - t0, 2)))
    # print("Number of Rows [{}]".format(tot_row))


if __name__ == "__main__":
    main()
