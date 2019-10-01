import mysql.connector
import ast

USER = 'user'
DB = 'forecast'
PWD = 'userpwd'
HOST = 'localhost'

column_link = {}
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

    if not check_table_exist(dbc, 'data'):
        print('Creating DATA table...')
        c.execute("CREATE TABLE data ("
                  "id INT AUTO_INCREMENT PRIMARY KEY,"
                  "name VARCHAR(255),"
                  "start_dt DATETIME,"
                  "end_dt DATETIME,"
                  "duration INT,"
                  "type VARCHAR(100),"
                  "subtype VARCHAR(100),"
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
    else:
        print('Table DATA already present...')

    if not check_table_exist(dbc, 'dataset'):
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
    else:
        print('Table DATASET already present...')

# def loading_csv(path):


def read_dict():
    # dict.txt is a file containing the match between the column of the DB and the columns of the pandas Data Frame
    # DB_COL:DF_COL
    with open(BASE_PATH + 'dict.txt', 'r') as f:
        for line in f:
            (key, val) = line.split(':')
            column_link[key] = val.replace('\n', '')


def main():
    # Database Connector
    dbc = connect()

    # Reading the dictionary used for matching
    read_dict()

    # Creating Tables
    create_table(dbc)

    # Loading data on DB
    # loading_csv(path)


if __name__ == "__main__":
    main()
