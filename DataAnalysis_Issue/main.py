from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
import sys

# Environment commands
max_features = 100


def merge(dataset):
    issue = pd.read_csv('data/' + dataset + '/issue.csv', nrows=None, parse_dates=True)
    # issue = issue.drop('description', axis=1)
    issue = issue.drop('created_date', axis=1)
    issue = issue.drop('updated_date', axis=1)
    issue = issue.drop('resolved_date', axis=1)

    change_set = pd.read_csv('data/' + dataset + '/change_set.csv', nrows=None, parse_dates=True)
    change_set = change_set.drop('committed_date', axis=1)
    change_set = change_set.drop('is_merge', axis=1)

    change_link = pd.read_csv('data/' + dataset + '/change_set_link.csv', nrows=None, parse_dates=True)

    fix_version = pd.read_csv('data/' + dataset + '/issue_fix_version.csv', nrows=None, parse_dates=True)

    code_change = pd.read_csv('data/' + dataset + '/code_change.csv', nrows=None, parse_dates=True)
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
    issue = pd.read_csv('data/' + dataset + '/issue.csv', nrows=None, parse_dates=True)

    starting_shape = issue.shape
    print('Starting shape:\t{}'.format(starting_shape))

    # issue = issue[(issue['resolution'] == 'Fixed')]

    issue['open_dt'] = pd.to_datetime(issue['created_date_zoned'])
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
    issue = issue.drop('time', axis=1)

    issue = issue.dropna()
    issue = remove_outliers(issue)

    prior = ['Critical', 'Major', 'Blocker', 'Minor', 'Trivial']
    type = ['Bug', 'Sub-task', 'Improvement', 'Test', 'Task', 'Wish', 'New Feature']

    issue.priority.replace(prior, [1,2,3,4,5], inplace=True)
    issue.type.replace(type, [1,2,3,4,5,6,7], inplace=True)

    issue['severity'] = issue.priority * issue.type

    component = pd.read_csv('data/' + dataset + '/issue_component.csv', nrows=None, parse_dates=True)
    issue_component = pd.merge(issue, component, on='issue_id', how='left')

    convert(issue_component, 'component')

    print('Starting recognition...', end='')
    # vectorized = word_recognition(issue_component['summary'])
    # vectorized = word_recognition(issue_component['summary'] + ' ' + issue_component['description'])
    issue_component = issue_component.drop('summary', axis=1)
    issue_component = issue_component.drop('description', axis=1)
    # issue_final = pd.concat([issue_component, vectorized], axis=1)
    issue_final = issue_component
    print(' OK')

    issue_final = issue_final.drop('issue_id', axis=1)

    print('Ending shape:\t{}'.format(issue_final.shape))
    print('\nData removed:\t{}%\n'.format(round((starting_shape[0] - issue_final.shape[0]) * 100 / starting_shape[0], 1)))

    print('Export...', end='')
    issue_final.to_csv(dataset + '_duration.csv', index=None)
    print(' OK')


def issue_count_forecast_file(dataset):
    issue = pd.read_csv('data/' + dataset + '/issue.csv', nrows=None, parse_dates=True)

    print('Starting shape:\t{}'.format(issue.shape))

    issue['open_dt'] = pd.to_datetime(issue['created_date_zoned'])
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
    # issue = issue.drop('type', axis=1)
    convert(issue, 'type')
    issue = issue.drop('priority', axis=1)
    # convert(issue, 'priority')
    issue = issue.drop('resolution', axis=1)
    issue = issue.drop('description', axis=1)
    issue['n'] = 0

    # issue_final = issue.groupby(by=['date', 'h', 'm'], as_index=True).count()
    # issue_final = issue.groupby(by=['date', 'type'], as_index=True).count()
    issue_final = issue.groupby(by=['date'], as_index=True).count()
    # issue_final = issue_final.reset_index()
    # issue_final = issue_final.set_index(['date', 'type'])['n'].unstack()
    issue_final = issue_final.fillna(0)
    issue_final = issue_final.reset_index()

    # issue_final['n'] = issue_final[0] + issue_final[1] + issue_final[2] + issue_final[3] + issue_final[4] + issue_final[5] + issue_final[6]
    # issue_final = issue.drop('type', axis=1)

    issue_final['date'] = pd.to_datetime(issue_final['date'])
    issue_final['month'] = issue_final['date'].dt.month
    # issue_final['year'] = issue_final['date'].dt.year
    # issue_final['day'] = issue_final['date'].dt.day
    issue_final['wday'] = issue_final['date'].dt.weekday
    # issue_final = issue_final.drop('type', axis=1)
    issue_final = issue_final.drop('date', axis=1)
    # issue_final = issue_final.drop('h', axis=1)
    # issue_final = issue_final.drop('m', axis=1)

    print('Ending shape:\t{}'.format(issue_final.shape))
    print('Export...', end='')
    issue_final = issue_final.reset_index()
    issue_final.to_csv(dataset + '_count.csv', index=None)
    print(' OK')


def word_recognition(df_cols):
    additional = frozenset([])

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
    # Remove outliers. Outliers defined as values greater than 99.5th percentile
    maxVal = np.percentile(df['n'], 75)
    minVal = np.percentile(df['n'], 25)
    # 8760 h == 365 days
    # 720 h == 30 days
    # 168 h == 7 days
    # 48 h == 2 days
    df = df[df['n'] <= maxVal]
    df = df[df['n'] >= minVal]
    return df


def main(dataset):
    # merge(dataset)
    issue_duration_forecast_file(dataset)
    # issue_count_forecast_file(dataset)


if __name__ == "__main__":
    main(sys.argv[1])
