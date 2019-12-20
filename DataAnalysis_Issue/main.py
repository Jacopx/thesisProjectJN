from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import sqlite3
import glob
import requests
import re


# Environment commands
max_features = 2000


def open_sqlite(dataset, table):
    conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')
    df = pd.read_sql_query('SELECT * FROM {};'.format(table), conn)
    return df


def merge(dataset):
    issue = open_sqlite(dataset, 'issue')
    # issue = issue.drop('description', axis=1)
    issue = issue.drop('created_date', axis=1)
    issue = issue.drop('updated_date', axis=1)
    issue = issue.drop('resolved_date', axis=1)

    change_set = open_sqlite(dataset, 'change_set')
    change_set = change_set.drop('committed_date', axis=1)
    change_set = change_set.drop('is_merge', axis=1)

    change_link = open_sqlite(dataset, 'change_set_link')

    fix_version = open_sqlite(dataset, 'issue_fix_version')

    code_change = open_sqlite(dataset, 'code_change')
    code_change = code_change.drop('file_path', axis=1)
    code_change = code_change.drop('old_file_path', axis=1)

    print('Merge', end='')

    issue_commit = pd.merge(issue, fix_version, on='issue_id', how='right')
    print('.', end='')
    issue_commit = pd.merge(issue_commit, change_link, on='issue_id', how='right')
    print('.', end='')
    issue_commit = pd.merge(issue_commit, change_set, on='commit_hash', how='left')
    print('.', end='')
    issue_commit = pd.merge(issue_commit, code_change, on='commit_hash', how='right')
    print(' OK')

    print('Export...', end='')
    issue_commit.to_csv(dataset + '-merged.csv', index=None)
    print(' OK')


def issue_duration_forecast_file(dataset):
    issue = open_sqlite(dataset, 'issue')

    starting_shape = issue.shape
    print('Starting shape:\t{}'.format(starting_shape))

    issue['open_dt'] = pd.to_datetime(issue['created_date_zoned'])
    # issue = issue.sort_values(by='created_date_zoned')
    issue = issue.drop('created_date', axis=1)
    issue = issue.drop('created_date_zoned', axis=1)

    issue['wd'] = issue['open_dt'].dt.weekday
    issue['h'] = issue['open_dt'].dt.hour
    # issue['m'] = issue['open_dt'].dt.minute

    issue = issue.drop('updated_date', axis=1)
    issue = issue.drop('updated_date_zoned', axis=1)

    issue['close_dt'] = pd.to_datetime(issue['resolved_date_zoned'])
    issue = issue.drop('resolved_date', axis=1)
    issue = issue.drop('resolved_date_zoned', axis=1)

    issue = issue.drop('status', axis=1)
    issue = issue.drop('assignee', axis=1)
    issue = issue.drop('assignee_username', axis=1)
    issue = issue.drop('reporter', axis=1)
    issue = issue.drop('reporter_username', axis=1)
    issue = issue.drop('resolution', axis=1)

    issue['time'] = issue['close_dt'] - issue['open_dt']
    issue = issue.drop('open_dt', axis=1)
    issue = issue.drop('close_dt', axis=1)
    issue['n'] = np.round(issue['time'].dt.total_seconds() / 60 / 60, 0)
    issue = issue.dropna()
    issue = remove_outliers(issue)
    issue['mov_avg'] = issue['n'].expanding().mean()
    issue['avg'] = issue['n'].mean()
    issue = issue.drop('time', axis=1)

    meansT = issue.groupby(by=['type'], as_index=True)['n'].mean()
    meansT = meansT.reset_index()

    issue = pd.merge(issue, meansT, on='type')
    issue = issue.rename(columns={'n_y': 'type_avg', 'n_x': 'n'})

    # meansP = issue.groupby(by=['priority'], as_index=True)['n'].mean()
    # meansP = meansP.reset_index()
    #
    # issue = pd.merge(issue, meansP, on='priority')
    # issue = issue.rename(columns={'n_y': 'prior_avg', 'n_x': 'n'})

    prior = ['Critical', 'Major', 'Blocker', 'Minor', 'Trivial']
    type = ['Bug', 'Sub-task', 'Improvement', 'Test', 'Task', 'Wish', 'New Feature']

    # issue.priority.replace(prior, [100, 80, 70, 30, 50], inplace=True)
    # issue.type.replace(type, [80, 20, 30, 20, 60, 20, 20], inplace=True)
    issue.priority.replace(prior, [1, 2, 3, 4, 5], inplace=True)
    issue.type.replace(type, [1, 2, 3, 4, 5, 6, 7], inplace=True)

    issue['typology'] = issue.priority * issue.type

    component = open_sqlite(dataset, 'issue_component')
    issue_component = pd.merge(issue, component, on='issue_id', how='left')

    convert(issue_component, 'component')

    print('Starting recognition...', end='')
    vectorized = word_recognition(issue_component['summary'])
    # vectorized = word_recognition(issue_component['summary'] + ' ' + issue_component['description'])
    issue_component = issue_component.drop('summary', axis=1)
    issue_component = issue_component.drop('description', axis=1)
    issue_final = pd.concat([issue_component, vectorized], axis=1)
    # issue_final = issue_component
    print(' OK')

    issue_final = issue_final.drop('issue_id', axis=1)

    print('Ending shape:\t{}'.format(issue_final.shape))
    print('\nData removed:\t{}%\n'.format(round((starting_shape[0] - issue_final.shape[0]) * 100 / starting_shape[0], 1)))

    print('Export...', end='')
    issue_final.to_csv('data/CSV/' + dataset + '-duration.csv', index=None)
    print(' OK')


def issue_forecast_file(dataset):
    # Get all commit by WEEK and YEAR
    date_component_change = make_component_change(dataset)
    date_component_change['date'] = pd.to_datetime(date_component_change['date'])
    date_component_change['w'] = date_component_change['date'].dt.week
    date_component_change['y'] = date_component_change['date'].dt.year
    date_component_change = date_component_change[(date_component_change['y'] >= 2012) & (date_component_change['y'] <= 2018)]
    date_component_change = date_component_change.drop('date', axis=1)


    # BAG OF WORD OF COMPONENT CHANGES
    # vectorized = word_recognition(date_component_change['component'])
    # date_component_change = date_component_change.drop('component', axis=1)
    # date_wb_component = pd.concat([date_component_change, vectorized], axis=1)
    # week_commit = date_wb_component.groupby(by=['w', 'y']).sum()
    week_commit = date_component_change.groupby(by=['w', 'y']).sum()

    # SENIORITY
    # seniority = aggregate_sen(dataset)
    # week_commit_seniority = pd.merge(week_commit, seniority, on=['w', 'y'])
    week_commit_seniority = week_commit

    issue = open_sqlite(dataset, 'issue')
    print('Starting shape:\t{}'.format(issue.shape))

    issue['open_dt'] = pd.to_datetime(issue['created_date'])
    issue['o_date'] = issue['open_dt'].dt.date
    issue['o_y'] = issue['open_dt'].dt.year
    issue['o_w'] = issue['open_dt'].dt.week

    issue['close_dt'] = pd.to_datetime(issue['resolved_date'])
    issue['c_date'] = issue['close_dt'].dt.date
    issue['c_y'] = issue['close_dt'].dt.year
    issue['c_w'] = issue['close_dt'].dt.week

    issue['time'] = issue['close_dt'] - issue['open_dt']
    issue['duration'] = np.round(issue['time'].dt.total_seconds() / 60 / 60, 0)

    # FILTER
    issue = filter(issue)

    # REMOVE OR UPDATE FEATURES
    issue = issue.drop(['open_dt', 'close_dt', 'time', 'duration'], axis=1)
    issue = issue.drop(['created_date', 'created_date_zoned', 'updated_date', 'updated_date_zoned', 'resolved_date', 'resolved_date_zoned'], axis=1)
    issue = issue.drop(['status', 'assignee', 'assignee_username', 'reporter', 'reporter_username'], axis=1)
    issue = issue.drop(['issue_id', 'summary', 'description', 'type'], axis=1)
    issue = issue.rename(columns={'priority': 'severity'})

    prior = {'Critical': 50.00, 'Major': 10.00, 'Blocker': 2.00, 'Minor': 1.00, 'Trivial': 0.50, 'Mixed': 0}
    prior_list = {'Critical', 'Major', 'Blocker', 'Minor', 'Trivial'}

    # ADDING EMPTY COLUMN
    issue['n'] = 0

    for st, sv in prior.items():
        print('Type: ' + str(st) + ':')
        dup_issue = issue.copy()
        if st != 'Mixed':
            dup_issue = dup_issue[(dup_issue['severity'] == st)]

        # MAKE DF COPY TO COMPUTE OPEN ISSUE
        open_issue = dup_issue.copy()
        if st == 'Mixed':
            open_issue.severity.replace(prior_list, [50.00, 10.00, 2.00, 1.00, 0.50], inplace=True)
        else:
            open_issue.severity.replace(st, sv, inplace=True)
        open_issue_count = make_issue_count(open_issue, 'o_')
        open_issue_sum = make_issue_sum(open_issue, 'o_')

        # MAKE DF COPY TO COMPUTE CLOSED ISSUE
        close_issue = dup_issue.copy()
        if st == 'Mixed':
            close_issue.severity.replace(prior_list, [-50.00, -10.00, -2.00, -1.00, -0.50], inplace=True)
        else:
            close_issue.severity.replace(st, -sv, inplace=True)
        close_issue_count = make_issue_count(close_issue, 'c_')
        close_issue_sum = make_issue_sum(close_issue, 'c_')

        # COMPUTE SUM OF SEVERITY
        issue_sum = pd.merge(open_issue_sum, close_issue_sum, on=['y', 'w'], how='outer')
        issue_sum = issue_sum.fillna(0)
        issue_sum['severity_diff'] = issue_sum['open_severity_sum'] + issue_sum['close_severity_sum']
        issue_sum['cumsum_severity'] = issue_sum['severity_diff'].cumsum()

        # COMPUTE COUNT OF ISSUE
        issue_count = pd.merge(open_issue_count, close_issue_count, on=['y', 'w'], how='outer')
        issue_count = issue_count.fillna(0)
        issue_count['issue_diff'] = issue_count['open_issue_count'] + issue_count['close_issue_count']
        issue_count['cumsum_issue'] = issue_count['issue_diff'].cumsum()

        # MERGE THE COUNT
        issue_cnt_sum = pd.merge(issue_sum, issue_count, on=['y', 'w'])
        issue_final = pd.merge(issue_cnt_sum, week_commit_seniority, on=['y', 'w'])

        issue_final['line_change'] = issue_final['line_change'].astype('int32')
        issue_final['commit_count'] = issue_final['commit_count'].astype('int32')

        # CAST
        for i in range(3, len(issue_final.columns)):
            issue_final[issue_final.columns[i]] = issue_final[issue_final.columns[i]].astype('int32')

        print('Ending shape:\t{}'.format(issue_final.shape))
        print('Export:')

        issue_finalC = issue_final.copy()
        issue_finalP = issue_final.copy()

        horizons = [1, 2, 4, 6, 8, 10, 12, 16, 20, 30, 40, 52]
        # horizons = [1, 4, 8]

        # EXPORT FOR DIFFERENT TIME HORIZONS
        for shift in horizons:
            print('Horizon: ' + str(shift) + '.', end='')

            # COUNT
            issue_finalCx = issue_final.copy()
            issue_finalC = extracted_calculation2(issue_finalCx, 'cumsum_issue')
            issue_finalC[str(shift) + 'future'] = issue_finalC['cumsum_issue'].shift(-shift, fill_value=-1)
            issue_finalC = issue_finalC.head(-shift)
            issue_finalC = issue_finalC.drop('cumsum_issue', axis=1)
            issue_finalC = issue_finalC.rename(columns={str(shift) + "future": "n"})
            issue_finalC['n'] = np.round(issue_finalC['n'], 1)

            print('.', end='')

            # PRIORITY
            issue_finalPx = issue_final.copy()
            issue_finalP = extracted_calculation2(issue_finalPx, 'cumsum_severity')
            issue_finalP[str(shift) + 'future'] = issue_finalP['cumsum_severity'].shift(-shift, fill_value=-1)
            issue_finalP = issue_finalP.head(-shift)
            issue_finalP = issue_finalP.drop('cumsum_severity', axis=1)
            issue_finalP = issue_finalP.rename(columns={str(shift) + "future": "n"})
            issue_finalP['n'] = np.round(issue_finalP['n'], 1)

            print('.', end='')

            # EXPORT
            issue_finalC = issue_finalC.reset_index()
            issue_finalP = issue_finalP.reset_index()
            issue_finalC.to_csv('data/severity/' + dataset + '-' + st + '_count-' + str(shift) + '.csv', index=None)
            issue_finalP.to_csv('data/severity/' + dataset + '-' + st + '_prior-' + str(shift) + '.csv', index=None)
            print(' OK')


def make_component_change(dataset):
    conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')

    query = \
        """
            SELECT date, component, COUNT(DISTINCT commit_hash) as 'commit_count', SUM(line_change) as 'line_change', resolution, status
            FROM
             (  SELECT changes.commit_hash, component, date, line_change, resolution, status
                FROM issue_component,
                    (   SELECT change_set_link.issue_id, change.commit_hash, date, line_change, resolution, status
                        FROM issue, change_set_link,
                         (  SELECT change_set.commit_hash, DATE(committed_date) as 'date', SUM(sum_added_lines)-SUM(sum_removed_lines) AS 'line_change'
                            FROM code_change, change_set
                            WHERE change_set.commit_hash=code_change.commit_hash
                            GROUP BY change_set.commit_hash
                        ) as 'change'
                        WHERE issue.issue_id=change_set_link.issue_id AND change_set_link.commit_hash=change.commit_hash AND type='Bug'
                        ORDER BY issue.issue_id
                    ) as changes
                WHERE issue_component.issue_id=changes.issue_id
            )
            GROUP BY date, component;
        """

    return pd.read_sql_query(query, conn)


def make_component_change_clean(dataset):
    conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')

    query = \
        """
            SELECT date, component, COUNT(DISTINCT commit_hash) as 'commit_count', SUM(line_change) as 'line_change', resolution, status
            FROM
             (  SELECT changes.commit_hash, component, date, line_change, resolution, status
                FROM issue_component,
                    (   SELECT change_set_link.issue_id, change.commit_hash, date, line_change, resolution, status
                        FROM issue, change_set_link,
                         (  SELECT change_set.commit_hash, DATE(committed_date) as 'date', SUM(sum_added_lines)-SUM(sum_removed_lines) AS 'line_change'
                            FROM code_change, change_set
                            WHERE change_set.commit_hash=code_change.commit_hash
                            GROUP BY change_set.commit_hash
                        ) as 'change'
                        WHERE issue.issue_id=change_set_link.issue_id AND change_set_link.commit_hash=change.commit_hash
                        ORDER BY issue.issue_id
                    ) as changes
                WHERE issue_component.issue_id=changes.issue_id
            )
            GROUP BY date, component;
        """

    return pd.read_sql_query(query, conn)


def aggregate_sen(dataset):
    date = make_commit_dev_date(dataset)

    dev = make_sen_dev(dataset)
    dev['first'] = pd.to_datetime(dev['first'])
    dev['today'] = pd.datetime.today()

    dev['age'] = dev['today'] - dev['first']
    dev['age'] = np.round(dev['age'].dt.total_seconds() / 60 / 60 / 24, 0)
    dev['age'] = dev['age'].astype('int32')
    dev['seniority'] = dev['age'] * dev['commit_count'] # TYPE1
    # dev['seniority'] = dev['commit_count'] # TYPE2
    # dev['seniority'] = dev['age'] # TYPE3
    # dev['seniority'] = (0.6 * dev['age']) + (dev['commit_count'] * 0.4) # TYPE4
    dev = dev.drop(['today', 'first'], axis=1)

    df = pd.merge(date, dev, on=['author'])
    df = df.drop(['author', 'commit_count', 'age'], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df['y'] = df['date'].dt.year
    df['w'] = df['date'].dt.week
    final = df.groupby(by=['w', 'y']).sum()

    return final


def make_commit_dev_date(dataset):
    conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')

    query = \
        """
            SELECT author, DATE(committed_date) as 'date'
            FROM change_set;
        """

    return pd.read_sql_query(query, conn)


def make_sen_dev(dataset):
    conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')

    query = \
        """
            SELECT author, MIN(DATE(committed_date)) as 'first', COUNT(commit_hash) as 'commit_count'
            FROM change_set
            GROUP BY author
            ORDER BY COUNT(commit_hash) DESC;
        """

    return pd.read_sql_query(query, conn)


def make_issue_count(df, root):
    df = df.rename(columns={root + 'w': 'w'})
    df = df.groupby(by=['w'], as_index=True).count()
    df = df.drop('severity', axis=1)
    df = df.drop('resolution', axis=1)

    if root in 'o_':
        df = df.rename(columns={'n': 'open_issue_count'})
    else:
        df['n'] = -df['n']
        df = df.rename(columns={'n': 'close_issue_count'})

    return df


def make_issue_sum(df, root):
    df = df.rename(columns={root + 'w': 'w'})
    df = df.groupby(by=['w'], as_index=True).sum()
    df = df.drop('n', axis=1)

    if root in 'o_':
        df = df.rename(columns={'severity': 'open_severity_sum'})
    else:
        df = df.rename(columns={'severity': 'close_severity_sum'})

    return df


def filter(df_init):
    df = df_init.copy()
    df = df[(df['type'] == 'Bug')]
    df = df[((df['o_y'] >= 2012) & (df['o_y'] <= 2018) & (df['c_y'] >= 2012) & (df['c_y'] <= 2018)) | ((df['c_y'].isna() == True) & (df['o_y'] == 2017))]
    df = df[((df['status'] == 'Closed') & (df['resolution'] == 'Fixed')) | ((df['status'] == 'Resolved') & (df['resolution'] == 'Fixed')) | (df['status'] == 'Open')]
    return df


def filter2(df_init):
    df = df_init.copy()
    df = df[(df['type'] == 'Bug')]
    df = df[(df['close_dt'].isna() == False)]
    df = df[((df['status'] == 'Closed') & (df['resolution'] == 'Fixed')) | ((df['status'] == 'Resolved') & (df['resolution'] == 'Fixed'))]
    return df


def word_recognition(df_cols):
    # additional = frozenset(['add', 'allow', 'application', 'block', 'change', 'check', 'command', 'common', 'configuration', 'create', 'data',
    #                         'default', 'documentation', 'does', 'doesn', 'erasure', 'fail', 'failed', 'failing', 'fails', 'failure', 'findbugs', 'fix',
    #                         'improve', 'incorrect', 'instead', 'java', 'log', 'logs', 'message', 'new', 'non', 'read', 'remove', 'running', 'set',
    #                         'test', 'tests', 'use', 'using', 'work', 'wrong', 'client', 'error', 'client', 'container', 'file', 'files', 'update',
    #                         'make', 'support', 'branch', 'code', 'exception', 'api', 'build', 'path'])

    additional = frozenset()

    stop_words = text.ENGLISH_STOP_WORDS.union(additional)

    # vect = CountVectorizer(lowercase=True, preprocessor=None, analyzer='word', token_pattern=r'[a-zA-Z][a-zA-Z][a-zA-Z]+', stop_words=frozenset(stop_words), max_features=max_features)
    vect = CountVectorizer(lowercase=True, preprocessor=None, analyzer='word', stop_words=frozenset(stop_words), max_features=max_features)
    X = vect.fit_transform(df_cols)
    # print(vect.get_feature_names())
    return pd.DataFrame(X.todense(), columns=vect.get_feature_names())


def convert(df, col):
    dict_dest = {}
    temp_dict = dict(df[col])

    i = 0
    for t in temp_dict.values():
        if t not in dict_dest.keys():
            dict_dest[t] = i
            i = i + 1

    df[col].replace(dict_dest, inplace=True)


def remove_outliers(df, column):
    max_val = np.percentile(df[column], 95)
    df = df[df[column] <= max_val]

    min_val = np.percentile(df[column], 1)
    df = df[df[column] >= min_val]

    return df


def extracted_calculation(issue_start, column):
    issue_final = issue_start.copy()
    issue_final['exp_avg'] = issue_final[column].expanding().mean()
    issue_final['mov_avg2'] = issue_final[column].rolling(2).mean()
    issue_final['mov_avg2'] = issue_final['mov_avg2'].shift(1, fill_value=-1)
    issue_final['mov_avg4'] = issue_final[column].rolling(4).mean()
    issue_final['mov_avg4'] = issue_final['mov_avg4'].shift(1, fill_value=-1)
    issue_final = issue_final.tail(-1)

    issue_final['1before'] = issue_final[column].shift(1, fill_value=-1)
    issue_final['2before'] = issue_final[column].shift(2, fill_value=-1)
    issue_final['4before'] = issue_final[column].shift(4, fill_value=-1)

    return issue_final.tail(-4)


def extracted_calculation2(issue_start, column):
    issue_final = issue_start.copy()
    issue_final['exp_avg'] = issue_final[column].expanding().mean()
    issue_final['mov_avg2'] = issue_final[column].rolling(2).mean()
    issue_final['mov_avg2'] = issue_final['mov_avg2'].shift(1, fill_value=0)
    issue_final['mov_avg4'] = issue_final[column].rolling(4).mean()
    issue_final['mov_avg4'] = issue_final['mov_avg4'].shift(1, fill_value=0)

    issue_final['1before'] = issue_final[column].shift(1, fill_value=0)
    issue_final['2before'] = issue_final[column].shift(2, fill_value=0)
    issue_final['4before'] = issue_final[column].shift(4, fill_value=0)

    return issue_final


def data_distribution(dataset):
    conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')
    # query = 'SELECT DATE(created_date) as "date", COUNT(DISTINCT issue_id) as "c" FROM issue WHERE type="Bug" GROUP BY DATE(created_date);'
    query = """
            SELECT DATE(created_date) as "date", COUNT(DISTINCT issue.issue_id) as "c"
            FROM issue, issue_fix_version
            WHERE type="Bug" AND issue.issue_id=issue_fix_version.issue_id AND fix_version like '2.%'
            GROUP BY DATE(created_date);
            """
    df = pd.read_sql_query(query, conn)

    df['date'] = pd.to_datetime(df['date'])
    df['y'] = df['date'].dt.year
    df['w'] = df['date'].dt.week
    # df['n'] = df['c'].cumsum()
    df['n'] = df['c']

    plt.figure(figsize=(60, 20))
    df['date'] = df[['y', 'w']].astype(str).apply('-'.join, axis=1)
    sns.barplot(df['date'], df['n'], label='value', ci=None, color='green')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Date')
    plt.ylabel('n')
    # plt.minorticks_on()
    plt.grid(axis='both')
    plt.title('Data Distribution ' + dataset)
    plt.savefig('DataDistribution-' + dataset + '.png', dpi=240)
    plt.show()

    print(df['y'].unique())


def ludwig_export(dataset):
    # Get all commit by WEEK and YEAR
    date_component_change = make_component_change(dataset)
    date_component_change['date'] = pd.to_datetime(date_component_change['date'])
    date_component_change['w'] = date_component_change['date'].dt.week
    date_component_change['y'] = date_component_change['date'].dt.year
    date_component_change = date_component_change[
        (date_component_change['y'] >= 2012) & (date_component_change['y'] <= 2018)]
    date_component_change = date_component_change.drop('date', axis=1)

    week_commit = date_component_change.groupby(by=['w', 'y']).sum()


    issue = open_sqlite(dataset, 'issue')
    print('Starting shape:\t{}'.format(issue.shape))

    issue['open_dt'] = pd.to_datetime(issue['created_date'])
    issue['o_date'] = issue['open_dt'].dt.date
    issue['o_y'] = issue['open_dt'].dt.year
    issue['o_w'] = issue['open_dt'].dt.week

    issue['close_dt'] = pd.to_datetime(issue['resolved_date'])
    issue['c_date'] = issue['close_dt'].dt.date
    issue['c_y'] = issue['close_dt'].dt.year
    issue['c_w'] = issue['close_dt'].dt.week

    issue['time'] = issue['close_dt'] - issue['open_dt']
    issue['duration'] = np.round(issue['time'].dt.total_seconds() / 60 / 60, 0)

    # FILTER
    issue = filter(issue)

    # REMOVE OR UPDATE FEATURES
    issue = issue.drop(['open_dt', 'close_dt', 'time', 'duration'], axis=1)
    issue = issue.drop(['created_date', 'created_date_zoned', 'updated_date', 'updated_date_zoned', 'resolved_date',
                        'resolved_date_zoned'], axis=1)
    issue = issue.drop(['status', 'assignee', 'assignee_username', 'reporter', 'reporter_username'], axis=1)
    issue = issue.drop(['issue_id', 'summary', 'description', 'type'], axis=1)
    issue = issue.rename(columns={'priority': 'severity'})

    prior = ['Critical', 'Major', 'Blocker', 'Minor', 'Trivial']

    # ADDING EMPTY COLUMN
    issue['n'] = 0

    # MAKE DF COPY TO COMPUTE OPEN ISSUE
    open_issue = issue.copy()
    open_issue.severity.replace(prior, [50.00, 10.00, 2.00, 1.00, 0.50], inplace=True)
    open_issue_count = make_issue_count(open_issue, 'o_')
    open_issue_sum = make_issue_sum(open_issue, 'o_')

    # MAKE DF COPY TO COMPUTE CLOSED ISSUE
    close_issue = issue.copy()
    close_issue.severity.replace(prior, [-50.00, -10.00, -2.00, -1.00, -0.50], inplace=True)
    close_issue_count = make_issue_count(close_issue, 'c_')
    close_issue_sum = make_issue_sum(close_issue, 'c_')

    # COMPUTE SUM OF SEVERITY
    issue_sum = pd.merge(open_issue_sum, close_issue_sum, on=['y', 'w'], how='outer')
    issue_sum = issue_sum.fillna(0)
    issue_sum['severity_diff'] = issue_sum['open_severity_sum'] + issue_sum['close_severity_sum']
    issue_sum['cumsum_severity'] = issue_sum['severity_diff'].cumsum()

    # COMPUTE COUNT OF ISSUE
    issue_count = pd.merge(open_issue_count, close_issue_count, on=['y', 'w'], how='outer')
    issue_count = issue_count.fillna(0)
    issue_count['issue_diff'] = issue_count['open_issue_count'] + issue_count['close_issue_count']
    issue_count['cumsum_issue'] = issue_count['issue_diff'].cumsum()

    # MERGE THE COUNT
    issue_cnt_sum = pd.merge(issue_sum, issue_count, on=['y', 'w'])
    issue_final = pd.merge(issue_cnt_sum, week_commit, on=['y', 'w'])

    issue_final['line_change'] = issue_final['line_change'].astype('int32')
    issue_final['commit_count'] = issue_final['commit_count'].astype('int32')

    # CAST
    for i in range(3, len(issue_final.columns)):
        issue_final[issue_final.columns[i]] = issue_final[issue_final.columns[i]].astype('int32')

    print('Ending shape:\t{}'.format(issue_final.shape))
    print('Export:')

    issue_finalC = issue_final.copy()
    issue_finalP = issue_final.copy()

    horizons = [1, 2, 4, 8]

    # EXPORT FOR DIFFERENT TIME HORIZONS
    for shift in horizons:
        print('Horizon: ' + str(shift) + '.', end='')

        # COUNT
        issue_finalCx = issue_final.copy()
        issue_finalC = extracted_calculation(issue_finalCx, 'cumsum_issue')
        issue_finalC[str(shift) + 'future'] = issue_finalC['cumsum_issue'].shift(-shift, fill_value=-1)
        issue_finalC = issue_finalC.head(-shift)
        issue_finalC = issue_finalC.drop('cumsum_issue', axis=1)
        issue_finalC = issue_finalC.rename(columns={str(shift) + "future": "n"})
        issue_finalC['n'] = np.round(issue_finalC['n'], 0)

        print('.', end='')

        # PRIORITY
        issue_finalPx = issue_final.copy()
        issue_finalP = extracted_calculation(issue_finalPx, 'cumsum_severity')
        issue_finalP[str(shift) + 'future'] = issue_finalP['cumsum_severity'].shift(-shift, fill_value=-1)
        issue_finalP = issue_finalP.head(-shift)
        issue_finalP = issue_finalP.drop('cumsum_severity', axis=1)
        issue_finalP = issue_finalP.rename(columns={str(shift) + "future": "n"})
        issue_finalP['n'] = np.round(issue_finalP['n'], 1)

        print('.', end='')

        # EXPORT
        issue_finalC = issue_finalC.reset_index()
        issue_finalP = issue_finalP.reset_index()
        issue_finalC.to_csv('data/ludwig/' + dataset + '-count' + str(shift) + '.csv', index=None)
        issue_finalP.to_csv('data/ludwig/' + dataset + '-prior' + str(shift) + '.csv', index=None)
        print(' OK')


def version_forecast_file(dataset):
    versions = ['0', '1', '2', '3', '4', '5', '6']

    for version in versions:
        try:

            # Get all commit by WEEK and YEAR
            date_component_change = make_component_change_clean(dataset)
            date_component_change['date'] = pd.to_datetime(date_component_change['date'])
            date_component_change['w'] = date_component_change['date'].dt.strftime('%Y') + '-' + date_component_change['date'].dt.strftime('%W')
            date_component_change = date_component_change.drop('date', axis=1)

            week_commit = date_component_change.groupby(by=['w']).sum()

            conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')

            query = """
                    SELECT * 
                    FROM issue, issue_fix_version
                    WHERE issue.issue_id=issue_fix_version.issue_id AND fix_version like '{}%' 
                    """.format(version)

            issue = pd.read_sql_query(query, conn)

            print('Starting shape:\t{}'.format(issue.shape))

            issue['open_dt'] = pd.to_datetime(issue['created_date_zoned'])
            issue['o_w'] = issue['open_dt'].dt.strftime('%Y') + '-' + issue['open_dt'].dt.strftime('%W')

            issue['close_dt'] = pd.to_datetime(issue['resolved_date_zoned'])
            issue['c_w'] = issue['close_dt'].dt.strftime('%Y') + '-' + issue['close_dt'].dt.strftime('%W')

            issue['time'] = issue['close_dt'] - issue['open_dt']
            issue['duration'] = np.round(issue['time'].dt.total_seconds() / 60 / 60, 0)

            # FILTER
            issue = filter2(issue)

            min_date = issue['open_dt'].min().date()
            max_date = issue['close_dt'].max().date()

            # REMOVE OR UPDATE FEATURES
            issue = issue.drop(['open_dt', 'close_dt', 'time', 'duration'], axis=1)
            issue = issue.drop(['created_date', 'created_date_zoned', 'updated_date', 'updated_date_zoned', 'resolved_date', 'resolved_date_zoned'], axis=1)
            issue = issue.drop(['status', 'assignee', 'assignee_username', 'reporter', 'reporter_username'], axis=1)
            issue = issue.drop(['issue_id', 'summary', 'description', 'type'], axis=1)
            issue = issue.drop(['fix_version'], axis=1)
            issue = issue.rename(columns={'priority': 'severity'})

            prior_list = ['Critical', 'Major', 'Blocker', 'Minor', 'Trivial']

            # ADDING EMPTY COLUMN
            issue['n'] = 0

            # GET ALL THE WEEK IN THE TIME SLICE
            all_date = pd.date_range(start=min_date, end=max_date, freq='D')
            all_date = pd.DataFrame(all_date)
            all_date['w'] = all_date[0].dt.strftime('%Y') + '-' + all_date[0].dt.strftime('%W')
            all_date = all_date.drop([0], axis=1)
            all_date = all_date.drop_duplicates(keep='last')

            # MAKE DF COPY TO COMPUTE OPEN ISSUE
            open_issue = issue.copy()
            open_issue.severity.replace(prior_list, [50.00, 10.00, 2.00, 1.00, 0.50], inplace=True)
            open_issue_count = make_issue_count(open_issue, 'o_')
            open_issue_sum = make_issue_sum(open_issue, 'o_')

            # MAKE DF COPY TO COMPUTE CLOSED ISSUE
            close_issue = issue.copy()
            close_issue.severity.replace(prior_list, [-50.00, -10.00, -2.00, -1.00, -0.50], inplace=True)
            close_issue_count = make_issue_count(close_issue, 'c_')
            close_issue_sum = make_issue_sum(close_issue, 'c_')

            print('Opened: {}'.format(open_issue_sum.sum().values[0]))
            print('Closed: {}\n'.format(close_issue_sum.sum().values[0]))
            print('Difference: {}\n'.format(close_issue_sum.sum().values[0] + open_issue_sum.sum().values[0]))

            # COMPUTE SUM OF SEVERITY
            issue_sum = pd.merge(open_issue_sum, close_issue_sum, on=['w'], how='outer')
            complete_sum = pd.merge(all_date, issue_sum, how='outer', on=['w'])
            issue_sum = complete_sum.fillna(0)
            issue_sum['severity_diff'] = issue_sum['open_severity_sum'] + issue_sum['close_severity_sum']
            issue_sum['cumsum_severity'] = issue_sum['severity_diff'].cumsum()

            # COMPUTE COUNT OF ISSUE
            issue_count = pd.merge(open_issue_count, close_issue_count, on=['w'], how='outer')
            complete_count = pd.merge(all_date, issue_count, how='outer', on=['w'])
            issue_count = complete_count.fillna(0)
            issue_count['issue_diff'] = issue_count['open_issue_count'] + issue_count['close_issue_count']
            issue_count['cumsum_issue'] = issue_count['issue_diff'].cumsum()

            # MERGE THE COUNT
            issue_cnt_sum = pd.merge(issue_sum, issue_count, on=['w'])
            issue_final = pd.merge(issue_cnt_sum, week_commit, on=['w'], how='outer')
            issue_final = issue_final.fillna(0)

            issue_final['line_change'] = issue_final['line_change'].astype('int32')
            issue_final['commit_count'] = issue_final['commit_count'].astype('int32')

            # CAST
            for i in range(3, len(issue_final.columns)):
                issue_final[issue_final.columns[i]] = issue_final[issue_final.columns[i]].astype('int32')

            print('Ending shape:\t{}'.format(issue_final.shape))
            print('Export:')

            issue_finalC = issue_final.copy()
            issue_finalP = issue_final.copy()

            horizons = [1, 2, 4, 6, 8, 10, 12, 16, 20, 30, 40, 52]
            # horizons = [1, 4, 8]
            # horizons = [1]

            # EXPORT FOR DIFFERENT TIME HORIZONS
            for shift in horizons:
                print('Horizon: ' + str(shift) + '.', end='')

                # COUNT
                issue_finalCx = issue_final.copy()
                issue_finalC = extracted_calculation2(issue_finalCx, 'cumsum_issue')
                issue_finalC[str(shift) + 'future'] = issue_finalC['cumsum_issue'].shift(-shift, fill_value=-1)
                issue_finalC = issue_finalC.head(-shift)
                issue_finalC = issue_finalC.drop('cumsum_issue', axis=1)
                issue_finalC = issue_finalC.drop('w', axis=1)
                issue_finalC = issue_finalC.rename(columns={str(shift) + "future": "n"})
                # issue_finalC['y'] = issue_finalC['w'].str.split('-', n = 0, expand=True)[0]
                # issue_finalC['w'] = issue_finalC['w'].str.split('-', n = 0, expand=True)[1]
                issue_finalC['n'] = np.round(issue_finalC['n'], 1)

                print('.', end='')

                # PRIORITY
                issue_finalPx = issue_final.copy()
                issue_finalP = extracted_calculation2(issue_finalPx, 'cumsum_severity')
                issue_finalP[str(shift) + 'future'] = issue_finalP['cumsum_severity'].shift(-shift, fill_value=-1)
                issue_finalP = issue_finalP.head(-shift)
                issue_finalP = issue_finalP.drop('cumsum_severity', axis=1)
                issue_finalP = issue_finalP.drop('w', axis=1)
                issue_finalP = issue_finalP.rename(columns={str(shift) + "future": "n"})
                # issue_finalP['y'] = issue_finalP['w'].str.split('-', n = 0, expand=True)[0]
                # issue_finalP['w'] = issue_finalP['w'].str.split('-', n = 0, expand=True)[1]
                issue_finalP['n'] = np.round(issue_finalP['n'], 1)

                print('.', end='')

                # EXPORT
                issue_finalC = issue_finalC.reset_index()
                issue_finalC = issue_finalC.drop('index', axis=1)
                issue_finalP = issue_finalP.reset_index()
                issue_finalP = issue_finalP.drop('index', axis=1)
                issue_finalC.to_csv('data/version/' + dataset + '-version_' + version + '_count-' + str(shift) + '.csv', index=None)
                issue_finalP.to_csv('data/version/' + dataset + '-version_' + version + '_prior-' + str(shift) + '.csv', index=None)
                print(' OK')
        except:
            e = sys.exc_info()[0]
            print(e)


def export_visualization(dataset):
    versions = ['0', '1', '2', '3', '4', '5', '6']

    for version in versions:
        try:
            conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')

            query = """
                    SELECT *
                    FROM issue, issue_fix_version
                    WHERE issue.issue_id=issue_fix_version.issue_id AND fix_version like '{}%' 
                    """.format(version)

            issue = pd.read_sql_query(query, conn)

            print('Starting shape:\t{}'.format(issue.shape))

            issue['open_dt'] = pd.to_datetime(issue['created_date_zoned'])
            issue['o_w'] = issue['open_dt'].dt.strftime('%Y') + '-' + issue['open_dt'].dt.strftime('%W')

            issue['close_dt'] = pd.to_datetime(issue['resolved_date_zoned'])
            issue['c_w'] = issue['close_dt'].dt.strftime('%Y') + '-' + issue['close_dt'].dt.strftime('%W')

            # FILTER
            issue = filter2(issue)

            min_date = issue['open_dt'].min().date()
            max_date = issue['close_dt'].max().date()

            # REMOVE OR UPDATE FEATURES
            issue = issue.drop(['open_dt', 'close_dt'], axis=1)
            issue = issue.drop(['created_date', 'created_date_zoned', 'updated_date', 'updated_date_zoned', 'resolved_date', 'resolved_date_zoned'], axis=1)
            issue = issue.drop(['status', 'assignee', 'assignee_username', 'reporter', 'reporter_username'], axis=1)
            issue = issue.drop(['issue_id', 'summary', 'description', 'type', 'fix_version'], axis=1)
            issue = issue.rename(columns={'priority': 'severity'})

            prior_list = ['Critical', 'Major', 'Blocker', 'Minor', 'Trivial']

            # ADDING EMPTY COLUMN
            issue['n'] = 0

            # GET ALL THE WEEK IN THE TIME SLICE
            all_date = pd.date_range(start=min_date, end=max_date, freq='D')
            all_date = pd.DataFrame(all_date)
            all_date['w'] = all_date[0].dt.strftime('%Y') + '-' + all_date[0].dt.strftime('%W')
            all_date = all_date.drop([0], axis=1)
            all_date = all_date.drop_duplicates(keep='last')

            # MAKE DF COPY TO COMPUTE OPEN ISSUE
            open_issue = issue.copy()
            open_issue.severity.replace(prior_list, [50.00, 10.00, 2.00, 1.00, 0.50], inplace=True)
            open_issue_count = make_issue_count(open_issue, 'o_')
            open_issue_sum = make_issue_sum(open_issue, 'o_')

            # MAKE DF COPY TO COMPUTE CLOSED ISSUE
            close_issue = issue.copy()
            close_issue.severity.replace(prior_list, [-50.00, -10.00, -2.00, -1.00, -0.50], inplace=True)
            close_issue_count = make_issue_count(close_issue, 'c_')
            close_issue_sum = make_issue_sum(close_issue, 'c_')

            print('Opened: {}'.format(open_issue_sum.sum().values[0]))
            print('Closed: {}\n'.format(close_issue_sum.sum().values[0]))
            print('Difference: {}'.format(close_issue_sum.sum().values[0] + open_issue_sum.sum().values[0]))

            # COMPUTE SUM OF SEVERITY
            issue_sum = pd.merge(open_issue_sum, close_issue_sum, on=['w'], how='outer')
            complete_sum = pd.merge(all_date, issue_sum, how='outer', on=['w'])
            issue_sum = complete_sum.fillna(0)
            issue_sum['severity_diff'] = issue_sum['open_severity_sum'] + issue_sum['close_severity_sum']
            issue_sum['cumsum_severity'] = issue_sum['severity_diff'].cumsum()
            issue_sum = issue_sum.drop(['severity_diff', 'open_severity_sum', 'close_severity_sum'], axis=1)

            # COMPUTE COUNT OF ISSUE
            issue_count = pd.merge(open_issue_count, close_issue_count, on=['w'], how='outer')
            complete_count = pd.merge(all_date, issue_count, how='outer', on=['w'])
            issue_count = complete_count.fillna(0)
            issue_count['issue_diff'] = issue_count['open_issue_count'] + issue_count['close_issue_count']
            issue_count['cumsum_issue'] = issue_count['issue_diff'].cumsum()
            issue_count = issue_count.drop(['issue_diff', 'open_issue_count', 'close_issue_count', 'c_w', 'o_w'], axis=1)

            # MERGE THE COUNT
            issue_cnt_sum = pd.merge(issue_sum, issue_count, on=['w'])

            print('Ending shape:\t{}\n'.format(issue_cnt_sum.shape))

            issue_finalP = issue_cnt_sum.copy()
            issue_finalP = issue_finalP.rename(columns={'cumsum_severity': 'v{}'.format(version)})
            issue_finalP = issue_finalP.drop('cumsum_issue', axis=1)
            print('LATEST: {}'.format(issue_finalP['v{}'.format(version)].tail(1).values[0]))
            issue_finalP = issue_finalP.reset_index()
            issue_finalP = issue_finalP.drop('index', axis=1)
            issue_finalP.to_csv('data/visual/' + dataset + '-version_' + version + '_prior-visual.csv', index=None)

            issue_finalC = issue_cnt_sum.copy()
            issue_finalC = issue_finalC.rename(columns={'cumsum_issue': 'v{}'.format(version)})
            issue_finalC = issue_finalC.drop('cumsum_severity', axis=1)
            print('LATEST: {}'.format(issue_finalC['v{}'.format(version)].tail(1).values[0]))
            issue_finalC = issue_finalC.reset_index()
            issue_finalC = issue_finalC.drop('index', axis=1)
            issue_finalC.to_csv('data/visual/' + dataset + '-version_' + version + '_count-visual.csv', index=None)

        except:
            e = sys.exc_info()
            print(e)


def version_visualization(dataset):
    conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')

    query = """
            SELECT fix_version as version, MIN(DATE(created_date)) as date_s
            FROM issue, issue_fix_version
            WHERE issue.issue_id=issue_fix_version.issue_id
            GROUP BY fix_version
            ORDER BY MIN(DATE(created_date));
            """

    version = pd.read_sql_query(query, conn)
    version['f_date'] = pd.to_datetime(version['date_s'])
    version['date'] = version['f_date'].dt.date
    version['y'] = version['f_date'].dt.year
    version['w'] = version['f_date'].dt.week

    version = version.drop(['f_date', 'date', 'date_s'], axis=1)

    version.to_csv('data/release/' + dataset + '.csv', index=None)


def plot_all(df_original, file):
    plt.figure(figsize=(40, 18))
    df = df_original.copy()
    df['n'] = df['n'].astype('int32')
    df['date'] = df[['y', 'w']].astype(str).apply('-'.join, axis=1)
    sns.pointplot(df['date'], df['n'], label='value', ci=None, markersize=0.01, color='green')
    plt.xticks(rotation='60')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('n')
    plt.grid(axis='both')
    plt.title('Data Distribution: ' + file[5:])
    plt.savefig('DataDistribution-' + file[5:] + '.png', dpi=240)
    plt.show()


def all_version_plot(dataset):
    export_visualization(dataset)
    m = pd.DataFrame(columns=['w'])
    vers = []
    for f in glob.glob('data/visual/' + dataset + '-version_*_prior-visual.csv'):
        vers.append('v' + f.split('_')[1])
        v = pd.read_csv(f)
        m = pd.merge(m, v, how='outer', on='w')

    m = m.fillna(0)

    m = m.sort_values('w')
    vers.append('w')
    vers.sort()
    f = m[vers]

    f.plot(figsize=(22, 10), x='w')
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('Severity')
    plt.grid(axis='both')
    plt.title('Data Distribution: ' + dataset)
    plt.savefig('DataDistributionMerged-' + dataset + '.png', dpi=240)
    plt.show()

    f.plot(figsize=(20, 13), title=('Data Distribution: ' + dataset), subplots=True, x='w', sharex=True, grid=True, sharey=True)
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('Severity')
    plt.savefig('DataDistributionSplitted-' + dataset + '.png', dpi=240)
    plt.show()


def get_releases(dataset, repos):
    dict = {}

    for i in range(1, 100):
        resp = requests.get('https://api.github.com/repos/' + repos + '/tags?page={}'.format(i))
        if resp.status_code != 200:
            # This means something went wrong.
            raise ApiError('GET /tags/ {}'.format(resp.status_code))
        else:
            if resp.json() != []:
                for tags in resp.json():
                    c_resp = requests.get(tags['commit']['url'])
                    if c_resp.status_code != 200:
                        # This means something went wrong.
                        raise ApiError('GET /commit/ {}'.format(c_resp.status_code))
                    else:
                        commit = c_resp.json()
                        dict[tags['name']] = commit['commit']['author']['date']
                        print('{} == {}'.format(tags['name'], commit['commit']['author']['date']))
            else:
                break

        print('{} =================='.format(i))

    with open('data/release/' + dataset + '.csv', 'w') as f:
        print('release,date', file=f)
        for i,v in dict.items():
            print(i, v, sep=',', file=f)


def all_version_plot_release(dataset, repos):
    # export_visualization(dataset)
    # get_releases(dataset, repos)

    # Join file from different version
    m = pd.DataFrame(columns=['w'])
    vers = []
    for f in glob.glob('data/visual/' + dataset + '-version_*_prior-visual.csv'):
        vers.append('v' + f.split('_')[1])
        v = pd.read_csv(f)
        m = pd.merge(m, v, how='outer', on='w')

    # Fill the Nan with zero
    m = m.fillna(0)

    # Sort value by date and reorder version
    m = m.sort_values('w')
    vers.append('w')
    vers.sort()
    f = m[vers]

    # Get data for release
    r = pd.read_csv('data/release/' + dataset + '.csv')
    r['date_p'] = pd.to_datetime(r['date'])
    r['w'] = r['date_p'].dt.strftime('%Y') + '-' + r['date_p'].dt.strftime('%W')
    r = r.drop_duplicates(subset=['release'])
    r = r.drop(['date_p', 'date'], axis=1)

    # Subset the data respect date and release name format
    r = r[(r['w'] >= f['w'].min()) & (r['w'] <= f['w'].max())]

    pattern = regex_pattern(dataset)

    p = re.compile(pattern)
    r = r[(r['release'].str.match(p))]

    colors = {0:'steelblue', 1:'darkorange', 2: 'green', 3:'darkred', 4:'dodgerblue', 5:'gray', 6:'aquamarine', 7:'violet'}

    x = pd.merge(f, r, how='left', on=['w'])
    x['count'] = 1
    x = x.sort_values('w')
    x['count'] = x['count'].cumsum()

    s = x[['release', 'count']]
    s = s.dropna()

    x = x.drop(['release', 'count'], axis=1)
    x = x.drop_duplicates()

    vr = []
    for index, row in s.iterrows():
        vr.append(get_color(dataset, row['release']))

    base = min(vr)

    fig = f.plot(figsize=(22, 10), x='w', color=colors.values())
    for index, row in s.iterrows():
        # if(row['release'])
        c = get_color(dataset, row['release'], base)
        fig.axvline(row['count'], linestyle='-.', label=row['release'], c=colors[c], linewidth=1.0)
    plt.legend()  # Graph labels
    plt.xlabel('Week')
    plt.ylabel('Severity')
    plt.grid(axis='both')
    plt.title('Data Distribution: ' + dataset)
    plt.savefig('DataDistributionMerged-' + dataset + '.png', dpi=240)
    plt.show()


def regex_pattern(dataset):
    if dataset == 'hadoop':
        regex_pattern = '[rel/]*release-[0-9].[0-9].0$'

    elif dataset == 'hbase':
        regex_pattern = '[0-9].[0-9][0]*.0$'

    elif dataset == 'cassandra':
        regex_pattern = 'cassandra-[0-9].[0-9].0$'

    elif dataset == 'hive':
        regex_pattern = '[rel/]*[storage-]*release-[0-9].[0-9].0$'

    elif dataset == 'maven':
        regex_pattern = 'maven-[0-9].[0-9].0$'

    elif dataset == 'lucene':
        # regex_pattern = 'releases/lucene[-solr]*/[0-9].[0-9][.0]*$'
        regex_pattern = 'releases/lucene[-solr]*/[0-9].0[.0]*$'

    return regex_pattern


def get_color(dataset, string, base=-1):
    if dataset == 'hadoop':
        val = int((string.split('.')[0]).split('-')[1])

    elif dataset == 'hbase':
        val = int((string.split('.')[0]))

    elif dataset == 'cassandra':
        val = int((string.split('.')[0]).split('-')[1])

    elif dataset == 'hive':
        if 'storage' in string:
            val = int((string.split('.')[0]).split('-')[2])
        else:
            val = int((string.split('.')[0]).split('-')[1])

    elif dataset == 'maven':
        val = int((string.split('.')[0]).split('-')[1])

    elif dataset == 'lucene':
        val = int((string.split('.')[0]).split('/')[2])

    if base == -1:
        return val
    else:
        return abs(base - val)


def main(dataset, repos):
    # merge(dataset)
    # issue_duration_forecast_file(dataset)
    # issue_forecast_file(dataset)
    # data_distribution(dataset)
    # ludwig_export(dataset)
    # version_forecast_file(dataset)
    # export_visualization(dataset)
    # version_visualization(dataset)
    all_version_plot(dataset)
    # all_version_plot_release(dataset, repos)
    # get_releases(dataset, repos)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
