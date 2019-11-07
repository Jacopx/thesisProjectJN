import general


def duration(dbc, dataset):
    sql = "SELECT eid, start_dt, end_dt " \
          "FROM event " \
          "WHERE event.dataset=('{}') " \
          "ORDER BY event.eid;".format(dataset)
    df = general.generate_df(dbc, sql)
    df['duration'] = df.apply(lambda row: calc_duration(row['start_dt'], row['end_dt']), axis=1)
    return df


def calc_duration(s, e): return e-s
