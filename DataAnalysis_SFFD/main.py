import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from datetime import datetime
import sys

unit = {}
loc = {}

subtypo = {'Potentially Life-Threatening': 1, 'Non Life-threatening': 2, 'Alarm': 3, 'Fire': 4}
subtypoR = {1: 'Potentially Life-Threatening', 2: 'Non Life-threatening', 3: 'Alarm', 4: 'Fire'}

typo = {'Medical Incident': 0, 'Outside Fire': 1, 'Alarms': 2, 'Citizen Assist / Service Call': 3,
        'Traffic Collision': 4, 'Other': 5, 'Structure Fire': 6, 'Smoke Investigation (Outside)': 7,
        'Electrical Hazard': 8, 'Elevator / Escalator Rescue': 9, 'Vehicle Fire': 10,
        'Gas Leak (Natural and LP Gases)': 11, 'Water Rescue': 12, 'Odor (Strange / Unknown)': 13, 'Fuel Spill': 14,
        'Train / Rail Incident': 15, 'Administrative': 16, 'Marine Fire': 17, 'Industrial Accidents': 18,
        'High Angle Rescue': 19, 'HazMat': 20, 'Explosion': 21, 'Confined Space / Structure Collapse': 22,
        'Assist Police': 23, 'Extrication / Entrapped (Machinery, Vehicle)': 24, 'Watercraft in Distress': 25,
        'Suspicious Package': 26, 'Train / Rail Fire': 27, 'Mutual Aid / Assist Outside Agency': 28,
        'Lightning Strike (Investigation)': 29, 'Aircraft Emergency': 30, 'Oil Spill': 31}
typoR = {0: 'Medical Incident', 1: 'Outside Fire', 2: 'Alarms', 3: 'Citizen Assist / Service Call',
         4: 'Traffic Collision', 5: 'Other', 6: 'Structure Fire', 7: 'Smoke Investigation (Outside)',
         8: 'Electrical Hazard', 9: 'Elevator / Escalator Rescue', 10: 'Vehicle Fire',
         11: 'Gas Leak (Natural and LP Gases)', 12: 'Water Rescue', 13: 'Odor (Strange / Unknown)', 14: 'Fuel Spill',
         15: 'Train / Rail Incident', 16: 'Administrative', 17: 'Marine Fire', 18: 'Industrial Accidents',
         19: 'High Angle Rescue', 20: 'HazMat', 21: 'Explosion', 22: 'Confined Space / Structure Collapse',
         23: 'Assist Police', 24: 'Extrication / Entrapped (Machinery, Vehicle)', 25: 'Watercraft in Distress',
         26: 'Suspicious Package', 27: 'Train / Rail Fire', 28: 'Mutual Aid / Assist Outside Agency',
         29: 'Lightning Strike (Investigation)', 30: 'Aircraft Emergency', 31: 'Oil Spill'}


def dot():
    print('.', end='')


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


def read_data(dest):
    print('Data read...', end='')
    path = 'data'  # use your path
    all_files = glob.glob(path + "/" + dest + ".csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, low_memory=False, parse_dates=True, error_bad_lines=False)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    print(' OK')
    return frame


def data_reduction(df):
    print('Remove columns...', end='')
    df = df[['unit_id', 'call_type', 'call_type_group', 'received_dt_tm', 'on_scene_dt_tm', 'available_dt_tm',
             'zipcodeof_incident', 'numberof_alarms', 'battalion', 'station_area', 'box', 'priority', 'location']]
    print(' OK')
    return df


def remove_outliers(df, col):
    print('Remove OUTLIERS', end='')
    # Remove outliers. Outliers defined as values greater than 99.5th percentile
    max_val = np.percentile(df[col], 99.5)
    dot()
    df = df[df[col] <= max_val]
    print('.. OK')
    return df


def fix_priority(df_in):
    print('Fixing priority ', end='')
    df = df_in.copy()
    df.priority.replace(['A', 'B', 'C', 'D', 'E'], ['2', '2', '2', '3', '3'], inplace=True)
    dot()

    df = df[(df.priority != 'I') & (df.priority != '1')]
    df.priority.dropna(axis=0, inplace=True)
    dot()

    df.priority.replace(['A', 'B', 'C', 'E'], ['2', '2', '2', '3'], inplace=True)

    df.priority.astype(int)
    print('. OK')

    return df


def parser(df):
    # received_dt_tm,on_scene_dt_tm,available_dt_tm
    print('Parser', end='')
    df['rec_dt'] = pd.to_datetime(df['received_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    dot()

    df['onscene_dt'] = pd.to_datetime(df['on_scene_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    dot()

    df['end_dt'] = pd.to_datetime(df['available_dt_tm'], format="%m/%d/%Y %I:%M:%S %p")
    print('. OK')


def feature_extraction(df):
    print('Feature extraction START', end='')
    # Extract date, month, hour of start
    df['rec_day'] = df['rec_dt'].dt.day
    dot()

    df['rec_month'] = df['rec_dt'].dt.month
    dot()

    df['rec_hour'] = df['rec_dt'].dt.hour
    dot()

    df['rec_day_of_week'] = df['rec_dt'].dt.weekday
    df['week'] = df['rec_dt'].dt.week
    print(' OK')

    print('Feature extraction END', end='')
    # Extract date, month, hour of end
    df['end_day'] = df['end_dt'].dt.day
    dot()

    df['end_month'] = df['end_dt'].dt.month
    dot()

    df['end_hour'] = df['end_dt'].dt.hour
    dot()

    df['end_day_of_week'] = df['end_dt'].dt.weekday
    print(' OK')

    print('Feature extraction DURATION', end='')
    # Extract duration
    d = df['end_dt'] - df['rec_dt']
    dot()

    d = d / 1000000000
    dot()

    df['duration'] = d.astype(int)
    print('. OK')

    print('Feature extraction RESPONSE TIME', end='')
    # Extract duration
    d = df['onscene_dt'] - df['rec_dt']
    dot()

    d = d / 1000000000
    dot()

    df['res_time'] = d.astype(int)
    print('. OK')


def remove_nan(df):
    df = df[['unit_id', 'call_type', 'call_type_group', 'rec_dt', 'onscene_dt', 'end_dt', 'zipcodeof_incident',
             'numberof_alarms', 'battalion', 'station_area', 'box', 'priority', 'location']]

    print('Removing NaN rows [' + str((df['end_dt'].isnull().sum() / df.size) * 100) + ']...', end='')
    df = df[np.isfinite(df['end_dt'])]
    print(' OK')

    print('Removing NaN rows [' + str((df['onscene_dt'].isnull().sum() / df.size) * 100) + ']...', end='')
    df = df[np.isfinite(df['onscene_dt'])]
    print(' OK')

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
    print('Correlation map', end='')
    corr = df.corr()
    dot()

    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, cmap='seismic', annot=True, linewidths=0.2, vmin=-1, vmax=1, square=False)
    dot()

    plt.title('Correlation Map')
    plt.ylabel('Features')
    plt.xlabel('Features')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/corr_map.png', dpi=300)
    dot()

    plt.show()
    print(' OK')


def distplot(df, col):
    print('Distribution of ' + col, end='')
    plt.figure(figsize=(12, 12))
    sub = df[(df[col] < 18000) & (df[col] > 0)]
    dot()

    sns.distplot(sub[col], kde=True)
    dot()

    plt.title('Distribution of ' + col)
    plt.ylabel('Density')
    plt.xlabel(col)
    dot()

    plt.minorticks_on()
    plt.show()
    print(' OK')


def weekday_hour(df_op):
    print('Hours distribution', end='')
    df = df_op[['rec_day_of_week', 'rec_hour', 'duration']]

    df_operations_day = pd.pivot_table(df[['rec_day_of_week', 'rec_hour', 'duration']],
                                       index=['rec_day_of_week', 'rec_hour'], aggfunc='count')
    df_operations_day = df_operations_day.sort_values(by=['rec_day_of_week'])
    dot()

    heatmap_data = pd.pivot_table(df_operations_day, values='duration', index='rec_day_of_week', columns='rec_hour')
    dot()

    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.1, vmin=0, square=True,
                cbar_kws={"orientation": "horizontal"})
    dot()

    plt.title('Operations over Day/Hours')
    plt.ylabel('Weekday')
    plt.xlabel('Hours')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/hours.png', dpi=300)
    plt.show()
    print(' OK')


def year_calendar(df_op):
    print('Years calendar', end='')
    df = df_op[['rec_day_of_week', 'week', 'priority']]

    df_operations_vehicle = pd.pivot_table(df, index=['rec_day_of_week', 'week'], aggfunc='count')
    dot()

    df_operations_vehicle2 = df_operations_vehicle.sort_values('rec_day_of_week', ascending=True)
    dot()

    heatmap_data = pd.pivot_table(df_operations_vehicle2, values='priority', columns='week', index='rec_day_of_week')
    plt.figure(figsize=(16, 4))
    sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.01, vmin=0, square=True,
                cbar_kws={"orientation": "horizontal"})
    dot()

    plt.title('Operations over Years')
    plt.ylabel('Weekdays')
    plt.xlabel('Week')
    plt.tight_layout()
    plt.savefig('plot/calendar.png', dpi=300)
    plt.show()
    print(' OK')


def op_over_month_station(df_op):
    print('Heatmap stations', end='')
    df = df_op[['station_area', 'rec_month', 'priority']]

    df_operations_day = pd.pivot_table(df, index=['station_area', 'rec_month'], aggfunc='count')
    dot()

    heatmap_data = pd.pivot_table(df_operations_day, values='priority', columns='station_area', index='rec_month')
    dot()

    plt.figure(figsize=(17, 6))
    sns.heatmap(heatmap_data, cmap="PuRd", linewidths=0.01, vmin=0, square=True, cbar_kws={"orientation": "horizontal"})
    dot()

    plt.title('Operations of different station over months')
    plt.xlabel('Station Area')
    plt.ylabel('Month')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig('plot/month_station.png', dpi=300)
    plt.show()
    print(' OK')


def hier_clust(df):
    print('Hierarchical clustering', end='')
    dfc = df[['priority', 'call_type_group']].copy()
    replace_dict(dfc, subtypoR, 'call_type_group')
    dot()

    dfc = dfc.groupby('call_type_group')
    priority = dfc.priority.value_counts().unstack().fillna(0)
    dot()

    priority_normalized = priority.div(priority.sum(axis=1), axis=0)
    h_cluster = sns.clustermap(priority_normalized, annot=True, cmap='Reds', fmt='g')

    h_cluster.fig.suptitle("Hierarchical Clustering of Call Type Group vs. Priority", size=25)
    ax = h_cluster.ax_heatmap
    ax.set_xlabel('Priority Level')
    ax.set_ylabel('Call Type Group')
    plt.savefig('plot/hier_type-prior.png', dpi=300)
    plt.show()
    print(' OK')


def export_csv(df, path):
    print('Exporting to CSV [' + path + ']..', end='')
    df.to_csv('data/' + path + '.csv', index=False)
    print('. OK')


def replace_dict(df, dict, col):
    print('Replacing value with dictionary...', end='')
    df.replace({col: dict}, inplace=True)
    print(' OK')


def main(path):
    t0 = time.time()
    print("Starting Analysis...\n")

    df = read_data(path)
    if 'RAW' in path:
        print("=== REDUCING DATASET ===\n")
        clean_col_names(df)
        df = data_reduction(df)
        export_csv(df, 'operationsSFFD_REDUCED')

    elif 'REDUCED' in path:
        print("=== CLEANING DATASET ===\n")
        parser(df)
        df = remove_nan(df)
        df = fix_priority(df)
        feature_extraction(df)
        df = remove_outliers(df, 'duration')
        df = remove_outliers(df, 'res_time')
        replace_dict(df, typo, 'call_type')
        replace_dict(df, subtypo, 'call_type_group')
        export_csv(df, 'operationsSFFD_CLEANED')

    else:
        print("=== COMPUTING DATASET ===\n")
        # Plots
        corr_map(df)
        distplot(df, 'duration')
        distplot(df, 'res_time')
        weekday_hour(df)
        year_calendar(df)
        op_over_month_station(df)
        hier_clust(df)

    print("\nTotal Time [{} s]".format(round(time.time() - t0, 2)))


if __name__ == "__main__":
    main(sys.argv[1])
