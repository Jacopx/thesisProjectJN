import sys
import numpy as np
import pandas as pd


def merge(dataset):
    issue = pd.read_csv('data/issue.csv', nrows=None, parse_dates=True)
    issue = issue.drop('description', axis=1)
    issue = issue.drop('created_date', axis=1)
    issue = issue.drop('updated_date', axis=1)
    issue = issue.drop('resolved_date', axis=1)

    change_set = pd.read_csv('data/change_set.csv', nrows=None, parse_dates=True)
    change_set = change_set.drop('committed_date', axis=1)
    change_set = change_set.drop('is_merge', axis=1)

    change_link = pd.read_csv('data/change_set_link.csv', nrows=None, parse_dates=True)

    fix_version = pd.read_csv('data/issue_fix_version.csv', nrows=None, parse_dates=True)

    code_change = pd.read_csv('data/code_change.csv', nrows=None, parse_dates=True)
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
    issue_commit.to_csv(dataset + '_issue.csv', index=None)
    print(' OK')


def main(dataset):
    merge(dataset)


if __name__ == "__main__":
    main(sys.argv[1])
