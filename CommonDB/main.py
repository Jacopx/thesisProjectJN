from sqlalchemy import create_engine
import mysql.connector
import pandas as pd
import time
import sys

# Docker
USER = 'user'
DB = 'forecast'
PWD = 'userpwd'
HOST = 'localhost'
PORT = '3306'

column_link = {}
link_column = {}
BASE_PATH = 'data_to_load/'


def connect():
    return mysql.connector.connect(host=HOST, port=PORT, user=USER, passwd=PWD, database=DB)


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

    tables = ['event', 'object', 'involved', 'info', 'locations']

    for t in tables:
        if check_table_exist(dbc, t):
            try:
                c.execute('DROP TABLE event, object, involved, info, location;')  # Syntax error in query
            except mysql.connector.Error as err:
                sys.stderr.write("Something went wrong: {}".format(err))

            print('Tables already present...')
            print('Drop tables...')
            break

    print('Creating EVENT table...')
    c.execute("CREATE TABLE event ("
              "eid VARCHAR(100) PRIMARY KEY,"
              "start_dt DATETIME,"
              "end_dt DATETIME);")

    print('Creating OBJECT table...')
    c.execute("CREATE TABLE object (id VARCHAR(100) PRIMARY KEY);")

    print('Creating INVOLVED table...')
    c.execute("CREATE TABLE involved (eid VARCHAR(100), "
              "id VARCHAR(100), "
              "PRIMARY KEY(eid, id), "
              "FOREIGN KEY (eid) REFERENCES event(eid), "
              "FOREIGN KEY (id) REFERENCES object(id));")

    print('Creating LOCATION table...')
    c.execute("CREATE TABLE location (id VARCHAR(100) PRIMARY KEY, "
              "latitude FLOAT, "
              "longitude FLOAT, "
              "FOREIGN KEY (id) REFERENCES object(id));")

    print('Creating INFO table...')
    c.execute("CREATE TABLE info (id VARCHAR(100) PRIMARY KEY, "
              "type VARCHAR(255), "
              "descr VARCHAR(1000), "
              "FOREIGN KEY (id) REFERENCES object(id));")


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


def check_event_exist(dbc, eid):
    c = dbc.cursor()
    sql = 'SELECT COUNT(*) FROM event WHERE eid=(%s);'
    c.execute(sql, [str(eid)])
    if c.fetchone()[0] == 1:
        c.close()
        return True


def check_obj_exist(dbc, id):
    c = dbc.cursor()
    sql = 'SELECT COUNT(*) FROM object WHERE id=(%s);'
    c.execute(sql, [str(id)])
    if c.fetchone()[0] == 1:
        c.close()
        return True


def check_info_exist(dbc, id):
    c = dbc.cursor()
    sql = 'SELECT COUNT(*) FROM info WHERE id=(%s);'
    c.execute(sql, [str(id)])
    if c.fetchone()[0] == 1:
        c.close()
        return True


def check_locat_exist(dbc, id):
    c = dbc.cursor()
    sql = 'SELECT COUNT(*) FROM location WHERE id=(%s);'
    c.execute(sql, [str(id)])
    if c.fetchone()[0] == 1:
        c.close()
        return True


def get_match(n):
    return link_column['id{}'.format(n)], link_column['type{}'.format(n)], link_column['descr{}'.format(n)], \
           link_column['latitude{}'.format(n)], link_column['longitude{}'.format(n)]


def load_to_db(name, df, dbc):
    c = dbc.cursor()

    print('Inserting data...')
    t0 = time.time()

    error = 0
    sql_event = 'INSERT INTO event(eid, start_dt, end_dt) VALUES (%s,%s,%s);'
    sql_obj = 'INSERT INTO object(id) VALUES (%s);'
    sql_invol = 'INSERT INTO involved(eid, id) VALUES (%s, %s);'
    sql_info = 'INSERT INTO info(id, type, descr) VALUES (%s, %s, %s);'
    sql_locat = 'INSERT INTO location(id, latitude, longitude) VALUES (%s, %s, %s);'

    for i, row in df.iterrows():
        t = list(row)

        if not check_event_exist(dbc, row[link_column['eid']]):
            try:
                c.execute(sql_event, [row[link_column['eid']], row[link_column['start_dt']], row[link_column['end_dt']]])  # Syntax error in query
            except mysql.connector.Error as err:
                error += 1
                sys.stderr.write("Something went wrong EVENT: {}\n{} = {}\n".format(err, i, t))

        for n in range(int(link_column['n'])):
            id, type, descr, lat, long = get_match(n)

            if id in 'ONE':
                if type in 'NONE':
                    obj_id = 'GPS#{}'.format(row[link_column['eid']])
                else:
                    obj_id = 'TYPE#{}'.format(row[link_column['eid']])
            elif id in 'UNIQUE':
                if type in 'NONE':
                    obj_id = 'GPS#{}-{}'.format(row[link_column['eid']], n)
                else:
                    obj_id = 'TYPE#{}-{}'.format(row[link_column['eid']], n)
            else:
                obj_id = row[id]

            if not check_obj_exist(dbc, obj_id):
                try:
                    c.execute(sql_obj, [obj_id])  # Syntax error in query
                except mysql.connector.Error as err:
                    error += 1
                    sys.stderr.write("Something went wrong OBJECT: {}\n{} = {}\n".format(err, i, t))

                try:
                    c.execute(sql_invol, [row[link_column['eid']], obj_id])  # Syntax error in query
                except mysql.connector.Error as err:
                    error += 1
                    sys.stderr.write("Something went wrong INVOLVED: {}\n{} = {}\n".format(err, i, t))

            if type not in 'NONE':
                if not check_info_exist(dbc, obj_id):
                    try:
                        c.execute(sql_info, [obj_id, row[type], row[descr]])  # Syntax error in query
                    except mysql.connector.Error as err:
                        error += 1
                        sys.stderr.write("Something went wrong INFO: {}\n{} = {}\n".format(err, i, t))

            if lat not in 'NONE':
                if not check_locat_exist(dbc, obj_id):
                    try:
                        c.execute(sql_locat, [obj_id, row[lat], row[long]])  # Syntax error in query
                    except mysql.connector.Error as err:
                        error += 1
                        sys.stderr.write("Something went wrong LOCAT: {}\n{} = {}\n".format(err, i, t))

        dbc.commit()

    print('\nError line #{}'.format(error))
    print("Execution time [{} s]".format(round(time.time()-t0, 2)))


def main(name, data, dict):
    # Database Connector
    dbc = connect()

    # Creating Tables
    create_table(dbc)

    # Reading the dictionary used for matching
    read_dict(dict)

    # Loading data on DB
    df = loading_csv(data)
    # rename_col(df)
    load_to_db(name, df, dbc)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
