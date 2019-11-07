import mysql.connector
import pandas as pd

# FreeNAS
USER = 'eis'
DB = 'forecast'
PWD = 'eisworld2019'
HOST = 'db.jacopx.me'
PORT = '3306'


def connect(): return mysql.connector.connect(host=HOST, port=PORT, user=USER, passwd=PWD, database=DB)


def generate_df(dbc, sql): return pd.read_sql(sql, dbc)


def export_csv(df, path):
    print('Exporting to CSV [' + path + ']..', end='')
    df.to_csv('data/' + path + '.csv', index=False)
    print('. OK')


def dot():
    print('.', end='')
