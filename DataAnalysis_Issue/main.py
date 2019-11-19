import sys
import numpy as np
import pandas as pd


def main(path):
    issue = pd.read_csv(path + '/issue.csv', nrows=None, parse_dates=True)
    issue = issue.drop('description', axis=1)


if __name__ == "__main__":
    main(sys.argv[1])
