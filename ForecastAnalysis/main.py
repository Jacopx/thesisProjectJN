import mysql.connector
import geopy.distance
import pandas as pd
from pandas import DataFrame
import time
import sys


# FreeNAS
USER = 'eis'
DB = 'forecastDev'
PWD = 'eisworld2019'
HOST = 'db.jacopx.me'
PORT = '3306'


def connect(): return mysql.connector.connect(host=HOST, port=PORT, user=USER, passwd=PWD, database=DB)


def distance(a, b): return geopy.distance.geodesic(a, b).m


def generate_df(dbc, sql): return pd.read_sql(sql, dbc)


def bs_distance(dbc, dataset):
    sql = "SELECT event.eid, involved.type, object.id as s_id, latitude as s_lat, longitude as s_long " \
          "FROM event, involved, object, location " \
          "WHERE event.eid=involved.eid " \
          "AND event.dataset=involved.dataset " \
          "AND involved.dataset=object.dataset " \
          "AND object.id=involved.id " \
          "AND object.id=location.id " \
          "AND event.dataset=('{}') " \
          "AND location.dataset=('{}') " \
          "AND involved.type=('{}') " \
          "AND object.type=('{}') " \
          "ORDER BY event.eid " \
          "LIMIT 6;".format(dataset, dataset, 'src', 'passive')
    df_src = generate_df(dbc, sql)
    print(df_src)

    sql = "SELECT event.eid, involved.type, object.id as e_id, latitude as e_lat, longitude as e_long " \
          "FROM event, involved, object, location " \
          "WHERE event.eid=involved.eid " \
          "AND event.dataset=involved.dataset " \
          "AND involved.dataset=object.dataset " \
          "AND object.id=involved.id " \
          "AND object.id=location.id " \
          "AND event.dataset=('{}') " \
          "AND location.dataset=('{}') " \
          "AND involved.type=('{}') " \
          "AND object.type=('{}') " \
          "ORDER BY event.eid " \
          "LIMIT 6;".format(dataset, dataset, 'dest', 'passive')
    df_dst = generate_df(dbc, sql)
    print(df_dst)

    df = pd.merge(df_dst, df_src, on=['eid'])
    df = df[['eid', 's_id', 's_lat', 's_long', 'e_id', 'e_lat', 'e_long']]
    df['distance'] = df.apply(lambda x: distance((x['s_lat'], x['s_long']), (x['e_lat'], x['e_long'])), axis=1)
    return df


def main(dataset):
    # Database Connector
    dbc = connect()

    df = bs_distance(dbc, dataset)
    print(df)


if __name__ == "__main__":
    main(sys.argv[1])
