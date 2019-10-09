import pandas.io.sql as psql
import mysql.connector
import geopy.distance
import pandas as pd
import time
import sys

# FreeNAS
USER = 'eis'
DB = 'forecast'
PWD = 'eisworld2019'
HOST = 'db.jacopx.me'
PORT = '33306'


def connect():
    return mysql.connector.connect(host=HOST, port=PORT, user=USER, passwd=PWD, database=DB)


def distance(a, b): return geopy.distance.geodesic(a, b).m


def to_pandas(dbc, sql_query):
    c = dbc.cursor()
    df = psql.frame_query(sql_query, c)
    c.close()
    return df


def main(dataset):
    # Database Connector
    dbc = connect()
    print(dbc)

    to_pandas(dbc, 'SELECT ')


if __name__ == "__main__":
    main(sys.argv[1])
