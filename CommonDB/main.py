from sqlalchemy import create_engine
import mysql.connector
import pandas as pd
import time
import sys

USER = 'user'
DB = 'forecast'
PWD = 'userpwd'
HOST = 'localhost'

# NAME = 'SFFD'
NAME = 'SFBS'

column_link = {}
link_column = {}
BASE_PATH = 'data_to_load/'


def connect():
    return mysql.connector.connect(host=HOST, user=USER, passwd=PWD, database=DB)


def check_table_exist(dbc, tablename):
    c = dbc.cursor()
    c.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = '{0}'
            """.format(tablename.replace('\'', '\'\'')))
    if c.fetchone()[0] == 1:
        c.close()
        return True


def create_table(dbc):
    c = dbc.cursor()

    # if check_table_exist(dbc, 'data'):
    #     print('Table DATA already present...')
    #     c.execute('DROP TABLE data')
    #     print('Drop DATA table...')
    #
    # print('Creating DATA table...')
    # c.execute("CREATE TABLE data ("
    #           "id INT AUTO_INCREMENT PRIMARY KEY,"
    #           "name VARCHAR(255),"
    #           "start_dt DATETIME,"
    #           "end_dt DATETIME,"
    #           "duration INT,"
    #           "type VARCHAR(100),"
    #           "feat1 VARCHAR(100),"
    #           "feat2 VARCHAR(100),"
    #           "feat3 VARCHAR(100),"
    #           "feat4 VARCHAR(100),"
    #           "feat5 VARCHAR(100),"
    #           "feat6 VARCHAR(100),"
    #           "feat7 VARCHAR(100),"
    #           "feat8 VARCHAR(100),"
    #           "feat9 VARCHAR(100),"
    #           "feat10 VARCHAR(100))")

    if check_table_exist(dbc, 'dataset'):
        print('Table DATASET already present...')
        c.execute('DROP TABLE dataset')
        print('Drop DATASET table...')

    print('Creating DATASET table...')
    c.execute("CREATE TABLE dataset ("
              "name VARCHAR(255) PRIMARY KEY,"
              "descr VARCHAR(100),"
              "first_dt DATETIME,"
              "last_dt DATETIME,"
              "feat1 VARCHAR(100),"
              "feat2 VARCHAR(100),"
              "feat3 VARCHAR(100),"
              "feat4 VARCHAR(100),"
              "feat5 VARCHAR(100),"
              "feat6 VARCHAR(100),"
              "feat7 VARCHAR(100),"
              "feat8 VARCHAR(100),"
              "feat9 VARCHAR(100),"
              "feat10 VARCHAR(100))")


def read_dict(dict):
    # dict.txt is a file containing the match between the column of the DB and the columns of the pandas Data Frame
    # DB_COL:DF_COL
    with open(BASE_PATH + dict, 'r') as f:
        for line in f:
            (val, key) = line.split(':')
            column_link[key.replace('\n', '')] = val.replace('\n', '')
            link_column[val.replace('\n', '')] = key.replace('\n', '')


def loading_csv(data):
    print('Reading CSV...', end='')
    df = pd.read_csv(BASE_PATH + data, index_col=None, low_memory=False, parse_dates=True, error_bad_lines=False)
    print(' OK')
    return df


def rename_col(df):
    df.rename(columns=column_link, inplace=True)


def load_to_db(df, dbc):
    c = dbc.cursor()

    sql = "INSERT INTO dataset(name, descr, feat1, feat2, feat3, feat4, feat5, feat6," \
          "feat7, feat8, feat9, feat10) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    val = (NAME, 'prova',
           link_column['feat1'], link_column['feat2'], link_column['feat3'], link_column['feat4'], link_column['feat5'],
           link_column['feat6'], link_column['feat7'], link_column['feat8'], link_column['feat9'], link_column['feat10'])

    c.execute(sql, val)
    dbc.commit()

    clean_dict = {}
    for (key, val) in link_column.items():
        if val in 'NULL':
            df[key] = 0
        else:
            clean_dict[key] = val

    # creating column list for insertion
    cols = "`,`".join([str(i) for i in list(clean_dict.keys())])

    # Removing NaN
    df = df[list(clean_dict.keys())]

    print('Inserting data...')
    t0 = time.time()
    # Insert DataFrame records one by one.
    sql = "INSERT INTO data (`name`,`" + cols + "`) VALUES (" + "%s," * (len(clean_dict)) + "%s)"
    error = 0

    for i, row in df.iterrows():
        t = list(row)
        t.insert(0, NAME)

        try:
            c.execute(sql, tuple(t))  # Syntax error in query
        except mysql.connector.Error as err:
            error += 1
            sys.stderr.write("Something went wrong: {}\n{} = {}\n".format(err, i, t))

        # Commit every 10000 tuples
        if i % 10000 == 0:
            print(i)
            dbc.commit()

        # Early stopper for debug
        # if i == 2000000:
        #     break

    dbc.commit()
    print('\nError line #{}'.format(error))
    print("Execution time [{} s]".format(round(time.time()-t0, 2)))


def main(data, dict):
    # Database Connector
    dbc = connect()

    # Creating Tables
    create_table(dbc)

    # Reading the dictionary used for matching
    read_dict(dict)

    # Loading data on DB
    df = loading_csv(data)
    rename_col(df)
    load_to_db(df, dbc)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
