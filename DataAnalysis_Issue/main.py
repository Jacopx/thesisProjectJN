import sys
import numpy as np
import pandas as pd


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


def issue_forecast_file(dataset):
    issue = pd.read_csv('data/' + dataset + '/issue.csv', nrows=None, parse_dates=True)

    issue['open_dt'] = pd.to_datetime(issue['created_date_zoned'])
    issue = issue.drop('created_date', axis=1)
    issue = issue.drop('created_date_zoned', axis=1)

    issue = issue.drop('updated_date', axis=1)
    issue = issue.drop('updated_date_zoned', axis=1)

    issue['close_dt'] = pd.to_datetime(issue['resolved_date_zoned'])
    issue = issue.drop('resolved_date', axis=1)
    issue = issue.drop('resolved_date_zoned', axis=1)

    issue = issue.drop('status', axis=1)
    issue = issue.drop('resolution', axis=1)
    issue = issue.drop('assignee', axis=1)
    issue = issue.drop('assignee_username', axis=1)
    issue = issue.drop('reporter', axis=1)
    issue = issue.drop('reporter_username', axis=1)

    issue['time'] = issue['close_dt'] - issue['open_dt']
    issue = issue.drop('open_dt', axis=1)
    issue = issue.drop('close_dt', axis=1)
    issue['minute'] = np.round(issue['time'].dt.total_seconds() / 60, 0)
    issue = issue.drop('time', axis=1)

    component = pd.read_csv('data/' + dataset + '/issue_component.csv', nrows=None, parse_dates=True)
    issue_component = pd.merge(issue, component, on='issue_id', how='left')

    print('Export...', end='')
    issue_component.to_csv(dataset + '_issue.csv', index=None)
    print(' OK')


def main(dataset):
    # merge(dataset)
    issue_forecast_file(dataset)


if __name__ == "__main__":
    main(sys.argv[1])
