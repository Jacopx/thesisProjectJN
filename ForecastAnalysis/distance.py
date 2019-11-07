import general
import pandas as pd
import geopy.distance


def distance(dbc, dataset, src, dest):
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
          "ORDER BY event.eid;".format(dataset, dataset, src, 'passive')
    df_src = general.generate_df(dbc, sql)

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
          "ORDER BY event.eid;".format(dataset, dataset, dest, 'passive')
    df_dst = general.generate_df(dbc, sql)

    df = pd.merge(df_dst, df_src, on=['eid'])
    df = df[['eid', 's_id', 's_lat', 's_long', 'e_id', 'e_lat', 'e_long']]
    df['distance'] = df.apply(lambda row: calc_distance((row['s_lat'], row['s_long']), (row['e_lat'], row['e_long'])), axis=1)
    return df


def calc_distance(a, b): return geopy.distance.geodesic(a, b).m
