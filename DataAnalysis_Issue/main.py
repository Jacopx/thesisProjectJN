from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
import sys
import sqlite3

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
    issue_commit.to_csv(dataset + '_merged.csv', index=None)
    print(' OK')


def issue_duration_forecast_file(dataset):
    issue = open_sqlite(dataset, 'issue')

    starting_shape = issue.shape
    print('Starting shape:\t{}'.format(starting_shape))

    # issue = issue[(issue['resolution'] == 'Fixed')]

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
    issue_final.to_csv('data/CSV/' + dataset + '_duration.csv', index=None)
    print(' OK')


def issue_count_mixed_forecast_file(dataset):
    conn = sqlite3.connect('data/SQLITE3/' + dataset + '.sqlite3')

    query = \
    """
        SELECT date, component, COUNT(DISTINCT commit_hash) as 'count_commit', SUM(line_change) as 'line_change'
        FROM
         (  SELECT changes.issue_id, changes.commit_hash, component, date, line_change
            FROM issue_component,
                (   SELECT issue_id, change.commit_hash, date, line_change
                    FROM change_set_link,
                     (  SELECT change_set.commit_hash, DATE(committed_date) as 'date', SUM(sum_added_lines)-SUM(sum_removed_lines) AS 'line_change'
                        FROM code_change, change_set
                        WHERE change_set.commit_hash=code_change.commit_hash
                        GROUP BY change_set.commit_hash
                    ) as 'change'
                    WHERE change_set_link.commit_hash=change.commit_hash
                    ORDER BY issue_id
                ) as changes
            WHERE issue_component.issue_id=changes.issue_id
        )
        GROUP BY date, component;
    """

    date_component_change = pd.read_sql_query(query, conn)
    date_component_change['date'] = pd.to_datetime(date_component_change['date'])
    date_component_change['w'] = date_component_change['date'].dt.week
    date_component_change = date_component_change.drop('date', axis=1)

    vectorized = word_recognition(date_component_change['component'])
    date_component_change = date_component_change.drop('component', axis=1)
    date_wb_component = pd.concat([date_component_change, vectorized], axis=1)
    week_commit = date_wb_component.groupby(by='w').sum()

    issue = open_sqlite(dataset, 'issue')
    print('Starting shape:\t{}'.format(issue.shape))

    issue['open_dt'] = pd.to_datetime(issue['created_date'])
    issue['date'] = issue['open_dt'].dt.date
    issue['y'] = issue['open_dt'].dt.year
    issue['w'] = issue['open_dt'].dt.week
    # issue['h'] = issue['open_dt'].dt.hour
    # issue['m'] = issue['open_dt'].dt.minute

    issue = issue.drop('created_date', axis=1)
    issue = issue.drop('created_date_zoned', axis=1)
    issue = issue.drop('updated_date', axis=1)
    issue = issue.drop('updated_date_zoned', axis=1)
    issue = issue.drop('resolved_date', axis=1)
    issue = issue.drop('resolved_date_zoned', axis=1)
    issue = issue.drop('open_dt', axis=1)
    issue = issue.drop('status', axis=1)
    issue = issue.drop('assignee', axis=1)
    issue = issue.drop('assignee_username', axis=1)
    issue = issue.drop('reporter', axis=1)
    issue = issue.drop('reporter_username', axis=1)
    issue = issue.drop('issue_id', axis=1)
    issue = issue.drop('summary', axis=1)
    issue = issue.drop('description', axis=1)
    issue = issue.drop('type', axis=1)

    issue = issue[(issue['y'] >= 2012) & (issue['y'] <= 2018)]

    issue = issue.dropna()
    # issue = remove_outliers(issue)

    prior = ['Critical', 'Major', 'Blocker', 'Minor', 'Trivial']
    type = ['Bug', 'Sub-task', 'Improvement', 'Test', 'Task', 'Wish', 'New Feature']

    issue.priority.replace(prior, [1.00, 0.80, 0.70, 0.30, 0.50], inplace=True)

    issue_sum = issue.groupby(by=['w'], as_index=True)['priority'].sum()
    issue_final = pd.merge(issue_sum, date_component_change, on='w')

    # TODO (Final shape): | week# | commit# | comp_1 | ... | comp_N | line_variation | SUM(severity) | Others...
    print('Ending shape:\t{}'.format(issue_final.shape))
    print('Export...', end='')
    issue_final = issue_final.reset_index()
    issue_final.to_csv('data/CSV/' + dataset + '_mixed_count.csv', index=None)
    print(' OK')


def issue_count_forecast_file(dataset):
    issue = open_sqlite(dataset, 'issue')

    print('Starting shape:\t{}'.format(issue.shape))

    issue['open_dt'] = pd.to_datetime(issue['created_date'])
    issue['date'] = issue['open_dt'].dt.date
    # issue['h'] = issue['open_dt'].dt.hour
    # issue['m'] = issue['open_dt'].dt.minute

    issue = issue.drop('created_date', axis=1)
    issue = issue.drop('created_date_zoned', axis=1)
    issue = issue.drop('updated_date', axis=1)
    issue = issue.drop('updated_date_zoned', axis=1)
    issue = issue.drop('resolved_date', axis=1)
    issue = issue.drop('resolved_date_zoned', axis=1)
    issue = issue.drop('open_dt', axis=1)
    issue = issue.drop('status', axis=1)
    issue = issue.drop('assignee', axis=1)
    issue = issue.drop('assignee_username', axis=1)
    issue = issue.drop('reporter', axis=1)
    issue = issue.drop('reporter_username', axis=1)
    issue = issue.drop('issue_id', axis=1)
    issue = issue.drop('summary', axis=1)
    issue = issue.drop('description', axis=1)
    issue = issue.drop('type', axis=1)
    # convert(issue, 'type')
    issue = issue.drop('priority', axis=1)
    # convert(issue, 'priority')
    issue = issue.drop('resolution', axis=1)
    # convert(issue, 'resolution')

    issue['n'] = 0

    # issue_final = issue.groupby(by=['date', 'h', 'm'], as_index=True).count()
    # issue_final = issue.groupby(by=['date', 'h'], as_index=True).count()
    issue_final = issue.groupby(by=['date'], as_index=True).count()

    issue_final = issue_final.fillna(0)
    issue_final = issue_final.reset_index()

    issue_final['date'] = pd.to_datetime(issue_final['date'])
    issue_final['month'] = issue_final['date'].dt.month
    # issue_final['year'] = issue_final['date'].dt.year
    # issue_final['day'] = issue_final['date'].dt.day
    issue_final['wday'] = issue_final['date'].dt.weekday
    issue_final = issue_final.drop('date', axis=1)

    issue_final['exp_avg'] = issue_final['n'].expanding().mean()
    issue_final['mov_avg2'] = issue_final['n'].rolling(2).mean()
    issue_final['mov_avg2'] = issue_final['mov_avg2'].shift(1, fill_value=-1)
    issue_final = issue_final.tail(-2)
    # issue_final['mov_avg7'] = issue_final['n'].rolling(7).mean()
    # issue_final['mov_avg7'] = issue_final['mov_avg7'].shift(1, fill_value=-1)
    # issue_final = issue_final.tail(-7)

    issue_final['avg'] = issue_final['n'].mean()

    # issue_final['1before'] = issue_final['n'].shift(1, fill_value=-1)
    # issue_final['2before'] = issue_final['n'].shift(2, fill_value=-1)
    # issue_final['5before'] = issue_final['n'].shift(5, fill_value=-1)
    issue_final['7before'] = issue_final['n'].shift(7, fill_value=-1)
    issue_final['14before'] = issue_final['n'].shift(14, fill_value=-1)
    # issue_final['30before'] = issue_final['n'].shift(30, fill_value=-1)
    issue_final = issue_final.tail(-14)

    print('Ending shape:\t{}'.format(issue_final.shape))
    print('Export...', end='')
    issue_final = issue_final.reset_index()
    issue_final.to_csv('data/CSV/' + dataset + '_count.csv', index=None)
    print(' OK')
    return issue_final


def word_recognition(df_cols):
    additional = frozenset(['add', 'allow', 'application', 'block', 'change', 'check', 'command', 'common', 'configuration', 'create', 'data',
                            'default', 'documentation', 'does', 'doesn', 'erasure', 'fail', 'failed', 'failing', 'fails', 'failure', 'findbugs', 'fix',
                            'improve', 'incorrect', 'instead', 'java', 'log', 'logs', 'message', 'new', 'non', 'read', 'remove', 'running', 'set',
                            'test', 'tests', 'use', 'using', 'work', 'wrong', 'client', 'error', 'client', 'container', 'file', 'files', 'update',
                            'make', 'support', 'branch', 'code', 'exception', 'api', 'build', 'path'])

    stop_words = text.ENGLISH_STOP_WORDS.union(additional)

    vect = CountVectorizer(lowercase=True, preprocessor=None, analyzer='word', token_pattern=r'[a-zA-Z][a-zA-Z][a-zA-Z]+', stop_words=frozenset(stop_words), max_features=max_features)
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


def remove_outliers(df):
    max_val = np.percentile(df['n'], 60)
    df = df[df['n'] <= max_val]

    min_val = np.percentile(df['n'], 5)
    df = df[df['n'] >= min_val]

    return df


def main(dataset):
    # merge(dataset)
    # issue_duration_forecast_file(dataset)
    issue_count_mixed_forecast_file(dataset)
    # issue_count_forecast_file(dataset)


if __name__ == "__main__":
    main(sys.argv[1])
