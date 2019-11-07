import general
import time
import pandas as pd
import numpy as np
from datetime import datetime
epoch = datetime.utcfromtimestamp(0)

SECONDS = 1440


def unix_time(dt): return (dt - epoch).total_seconds() / 60


def saturation(dbc, dataset, unit, start, dest, type, gap, csv_name):
    t0 = time.time()
    print('Get data...')

    sql = """
          SELECT distinct YEAR(start_dt) as year
          FROM event WHERE dataset='{}' 
          ORDER BY YEAR(start_dt);
    """.format(dataset)
    df_year = general.generate_df(dbc, sql)

    sql = "SELECT id, descr FROM info WHERE dataset='{}' AND type='{}';".format(dataset, 'STATION')
    df_station = general.generate_df(dbc, sql)

    sql = """
        SELECT id as ss FROM info
        WHERE dataset='{}' AND type='{}';    
    """.format(dataset, 'STATION')
    df_get_stations = general.generate_df(dbc, sql)

    station_size = pd.Series(df_station.descr.values, index=df_station.id).to_dict()
    ds = {list(df_get_stations.ss.unique())[i]: i for i in range(0, len(list(df_get_stations.ss.unique())))}
    m = [np.ndarray(shape=(1, SECONDS)) for i in range(0, len(ds))]

    for s in ds:
        i = ds[s]
        m[i][0].fill(int(station_size[s])-gap)

    print('Compute saturation events...')
    df = pd.DataFrame(columns=['date', 'station', 'saturation'])

    for year in df_year.year.unique():

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

        df_src = general.generate_df(dbc, sql)

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

            df_dest = general.generate_df(dbc, sql)
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
            general.export_csv(df, csv_name)

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
        if unit[1:] in ds:
            i = ds[unit[1:]]
        else:
            return
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
        if unit[1:] in ds:
            i = ds[unit[1:]]
        else:
            return
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
