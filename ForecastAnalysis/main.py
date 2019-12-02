import sys
import general
import distance
import duration
import saturation
import forecast

def main(dataset):
    # Database Connector
    dbc = general.connect()

    # if dataset in 'SFBS':
    #     df = distance.distance(dbc, dataset, 'src', 'dest')
    # else:
    #     df = distance.distance(dbc, dataset, 'start', 'location')
    # print(df)

    # if dataset in 'SFBS':
    #     df = duration.duration(dbc, dataset)
    # else:
    #     df = duration.duration(dbc, dataset)
    # print(df)

    # if dataset in 'SFBS':
    #     df = saturation.saturation(dbc, dataset, 'with', 'src', 'dest', 1, 3, 'SFBS1')
    #     df = saturation.saturation(dbc, dataset, 'with', 'src', 'dest', 2, 3, 'SFBS2')
    # else:
    #     df = saturation(dbc, dataset, 'act', 'start', None, 1, 0, 'SFFD')

    if dataset in 'SFBS':
        forecast.count_model('data/station70_forecast')
    elif dataset in 'SFFD':
        forecast.count_model('data/satur')
    elif dataset in 'ISSUE':
        # forecast.count_model('data/hadoop_mixed_count')
        # forecast.count_model('data/hadoop_mixed_prior')
        # forecast.count_model('data/hadoop_mixed_count4')
        # forecast.count_model('data/hadoop_mixed_prior4')
        forecast.count_model('data/hadoop_mixed_count8')
        forecast.count_model('data/hadoop_mixed_prior8')
        # forecast.count_model('data/hadoop_mixed_count16')
        # forecast.count_model('data/hadoop_mixed_prior16')

        # forecast.duration_model('data/hive_duration')
        # forecast.count_model('data/hive_count')
        # forecast.duration_model('data/hbase_duration')
        # forecast.count_model('data/hbase_count')
        # forecast.duration_model('data/jbpm_duration')
        # forecast.count_model('data/jbpm_count')
        # forecast.duration_model('data/maven_duration')
        # forecast.count_model('data/maven_count')
        # forecast.duration_model('data/lucene_duration')
        # forecast.count_model('data/lucene_count')


if __name__ == "__main__":
    main(sys.argv[1])
