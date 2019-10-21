import mysql.connector
import geopy.distance
import pandas as pd
from datetime import datetime
import time
import sys
import numpy as np
epoch = datetime.utcfromtimestamp(0)


# FreeNAS
USER = 'eis'
DB = 'forecast'
PWD = 'eisworld2019'
HOST = 'db.jacopx.me'
PORT = '3306'

SECONDS = 1440


def unix_time(dt): return (dt - epoch).total_seconds() / 60


def connect(): return mysql.connector.connect(host=HOST, port=PORT, user=USER, passwd=PWD, database=DB)


def calc_distance(a, b): return geopy.distance.geodesic(a, b).m


def calc_duration(s, e): return e-s


def generate_df(dbc, sql): return pd.read_sql(sql, dbc)


def export_csv(df, path):
    print('Exporting to CSV [' + path + ']..', end='')
    df.to_csv('data/' + path + '.csv', index=False)
    print('. OK')


def dot():
    print('.', end='')


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
    df_src = generate_df(dbc, sql)

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
    df_dst = generate_df(dbc, sql)

    df = pd.merge(df_dst, df_src, on=['eid'])
    df = df[['eid', 's_id', 's_lat', 's_long', 'e_id', 'e_lat', 'e_long']]
    df['distance'] = df.apply(lambda row: calc_distance((row['s_lat'], row['s_long']), (row['e_lat'], row['e_long'])), axis=1)
    return df


def duration(dbc, dataset):
    sql = "SELECT eid, start_dt, end_dt " \
          "FROM event " \
          "WHERE event.dataset=('{}') " \
          "ORDER BY event.eid;".format(dataset)
    df = generate_df(dbc, sql)
    df['duration'] = df.apply(lambda row: calc_duration(row['start_dt'], row['end_dt']), axis=1)
    return df


def saturation(dbc, dataset, unit, start, dest, type, gap, csv_name):
    t0 = time.time()
    print('Get data...')

    sql = """
          SELECT distinct YEAR(start_dt) as year
          FROM event WHERE dataset='{}' 
          ORDER BY YEAR(start_dt);
    """.format(dataset)
    df_year = generate_df(dbc, sql)

    sql = "SELECT id, descr FROM info WHERE dataset='{}' AND type='{}';".format(dataset, 'STATION')
    df_station = generate_df(dbc, sql)

    sql = """
        SELECT id as ss FROM info
        WHERE dataset='{}' AND type='{}';    
    """.format(dataset, 'STATION')
    df_get_stations = generate_df(dbc, sql)

    station_size = pd.Series(df_station.descr.values, index=df_station.id).to_dict()
    ds = {list(df_get_stations.ss.unique())[i]: i for i in range(0, len(list(df_get_stations.ss.unique())))}
    m = [np.ndarray(shape=(1, SECONDS)) for i in range(0, len(ds))]

    for s in ds:
        i = ds[s]
        m[i][0].fill(int(station_size[s])-gap)

    print('Compute saturation events...')
    df = pd.DataFrame(columns=['date', 'station', 'saturation'])

    for year in df_year.year.unique():

        # cursor = dbc.cursor()
        #
        # sql = """
        #         SELECT count(eid) as c
        #         FROM event
        #         WHERE dataset='{}' and year(start_dt)='{}';
        #     """.format(dataset, year)
        #
        # cursor.execute(sql)
        # c = int(cursor.fetchone()[0])
        # cursor.close()
        #
        # if c == 0:
        #     print('SKIP YEAR: {}'.format(year))
        #     continue

        sql = """
                SELECT involved.id as ss, event.eid, MONTH(start_dt) as month, DAY(start_dt) as day, start_dt, end_dt, n.id as id
                FROM event, involved, (select involved.dataset, eid, info.id
                                            from involved, info
                                            where info.dataset=involved.dataset and info.id=involved.id
                                                and info.descr='N' and involved.type='{}'
                                        ) as n,
                                        (select involved.dataset, eid, info.id as start
                                            from involved, info
                                            where info.dataset=involved.dataset and info.id=involved.id
                                                and involved.type='{}'
                                            ) as s
                WHERE event.dataset=involved.dataset and event.eid=involved.eid
                  and event.dataset='{}' and event.dataset=n.dataset and event.eid=n.eid and event.eid=s.eid
                  and event.dataset=s.dataset and involved.id=s.start and n.dataset=s.dataset and n.eid=s.eid
                  and year(start_dt)='{}'
                ORDER BY day;
            """.format(unit, start, dataset, year)

        df_src = generate_df(dbc, sql)

        t1 = time.time()
        print('Query1: {} s'.format(round(time.time() - t0, 2)))

        if dest is not None:
            sql = """
                    SELECT involved.id as ss, event.eid, MONTH(start_dt) as month, DAY(start_dt) as day, start_dt, end_dt, n.id as id
                    FROM event, involved, (select involved.dataset, eid, info.id
                                                from involved, info
                                                where info.dataset=involved.dataset and info.id=involved.id
                                                    and info.descr='N' and involved.type='{}'
                                            ) as n,
                                            (select involved.dataset, eid, info.id as start
                                                from involved, info
                                                where info.dataset=involved.dataset and info.id=involved.id
                                                    and involved.type='{}'
                                                ) as s
                    WHERE event.dataset=involved.dataset and event.eid=involved.eid
                      and event.dataset='{}' and event.dataset=n.dataset and event.eid=n.eid and event.eid=s.eid
                      and event.dataset=s.dataset and involved.id=s.start and n.dataset=s.dataset and n.eid=s.eid
                      and year(start_dt)='{}'
                    ORDER BY day;
                """.format(unit, dest, dataset, year)

            df_dest = generate_df(dbc, sql)
        print('Query2: {} s'.format(round(time.time() - t1, 2)))

        for month in range(1, 13):
            for day in range(1, 32):
                t1 = time.time()

                df_day = df_src[(df_src.month == month) & (df_src.day == day)]
                if df_day.ss.count() == 0:
                    print('SKIP DAY {}-{}-{}'.format(day, month, year))
                    continue
                else:
                    print('{}-{}-{}'.format(year, month, day))

                first = datetime.strptime('{}-{}-{} 00:00:00'.format(year, month, day), '%Y-%m-%d %H:%M:%S')
                offset = unix_time(first)
                df_day.apply(lambda r: sub(dataset, ds, m, r['ss'], r['start_dt'], r['end_dt'], r['id'], offset), axis=1)

                if dest is not None:
                    df_day_dest = df_dest[(df_dest.month == month) & (df_src.day == day)]
                    df_day_dest.apply(lambda r: add(dataset, ds, m, r['ss'], r['start_dt'], r['end_dt'], r['id'], offset), axis=1)

                for s in ds:
                    total = 0
                    i = ds[s]
                    for sec in range(0, SECONDS):
                        if evaluate(m[i][0][sec], type, int(station_size[s])):
                            total += 1
                    # print('\t{} ==> {} s # M:{} m:{} \t# Refill with: {}'.format(s, total, max(m[i][0]), min(m[i][0]), int(station_size[s])-gap))

                    df = df.append({'date': '{}-{}-{}'.format(year, month, day), 'station': s, 'saturation': total}, ignore_index=True)

                    m[i][0].fill(int(station_size[s])-gap)

                print('{} s'.format(round(time.time() - t1, 2)))
            export_csv(df, csv_name)

    print('Total: {} s'.format(round(time.time() - t0, 2)))
    return df


def sub(dataset, ds, matrix, ss, start, end, unit, offset):
    start = int(unix_time(start))
    end = int(unix_time(end))
    offset = int(offset)

    if dataset == 'SFBS':
        i = ds[ss]
        last = offset + SECONDS - start
    elif dataset == 'SFFD':
        if unit[0].isalpha():
            i = ds[unit[1:]]
        else:
            i = ds[unit]
        last = end-start

    for sec in range(0, last):
        p = start - offset + sec
        if p >= SECONDS:
            break
        matrix[i][0][p] -= 1


def add(dataset, ds, matrix, ss, start, end, unit, offset):
    start = int(unix_time(start))
    end = int(unix_time(end))
    offset = int(offset)

    if dataset == 'SFBS':
        i = ds[ss]
        last = offset + SECONDS - start
    elif dataset == 'SFFD':
        if unit[0].isalpha():
            i = ds[unit[1:]]
        else:
            i = ds[unit]
        last = end - start

    for sec in range(0, last):
        p = start - offset + sec
        if p >= SECONDS:
            break
        matrix[i][0][p] += 1


def evaluate(v, type, s):
    if type == 1:
        return v <= 0
    else:
        return v >= s


def main(dataset):
    # Database Connector
    dbc = connect()

    # if dataset in 'SFBS':
    #     df = distance(dbc, dataset, 'src', 'dest')
    # else:
    #     df = distance(dbc, dataset, 'start', 'location')
    # print(df)
    #
    # if dataset in 'SFBS':
    #     df = duration(dbc, dataset)
    # else:
    #     df = duration(dbc, dataset)
    # print(df)

    if dataset in 'SFBS':
        df = saturation(dbc, dataset, 'with', 'src', 'dest', 1, 3, 'SFBS1')
        export_csv(df, 'SFBS1')
        df = saturation(dbc, dataset, 'with', 'src', 'dest', 2, 3, 'SFBS2')
        export_csv(df, 'SFBS2')
    else:
        df = saturation(dbc, dataset, 'act', 'start', None, 1, 0, 'SFFD')
        export_csv(df, 'SFFD')


if __name__ == "__main__":
    main(sys.argv[1])
