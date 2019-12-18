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
        # FORECAST
        # forecast.model_randomforest('data/hadoop-version_2_prior-1')
        # forecast.model_keras_lstm('data/hadoop-version_2_prior-1')
        # forecast.model_ludwig('data/hadoop-version_2_prior-1')

        # forecast.model_keras_nn('data/hadoop-version_0_prior-1')
        # forecast.model_keras_nn('data/hadoop-version_0_prior-4')
        # forecast.model_keras_nn('data/hadoop-version_0_prior-8')

        # forecast.model_keras_nn('data/hadoop-version_1_prior-1')
        # forecast.model_keras_nn('data/hadoop-version_1_prior-4')
        # forecast.model_keras_nn('data/hadoop-version_1_prior-8')

        # forecast.model_keras_nn('data/hadoop-version_2_prior-1')
        # forecast.model_keras_nn('data/hadoop-version_2_prior-4')
        # forecast.model_keras_nn('data/hadoop-version_2_prior-8')

        # forecast.model_keras_nn('data/hadoop-version_3_prior-1')
        # forecast.model_keras_nn('data/hadoop-version_3_prior-4')
        # forecast.model_keras_nn('data/hadoop-version_3_prior-8')

        forecast.model_keras_nn('data/hadoop-version_0_prior-visual')
        forecast.model_keras_nn('data/hadoop-version_1_prior-visual')
        forecast.model_keras_nn('data/hadoop-version_2_prior-visual')
        forecast.model_keras_nn('data/hadoop-version_3_prior-visual')

        forecast.model_keras_nn('data/hbase-version_0_prior-visual')
        forecast.model_keras_nn('data/hbase-version_1_prior-visual')
        forecast.model_keras_nn('data/hbase-version_2_prior-visual')

        forecast.model_keras_nn('data/hive-version_0_prior-visual')
        forecast.model_keras_nn('data/hive-version_1_prior-visual')
        forecast.model_keras_nn('data/hive-version_2_prior-visual')
        forecast.model_keras_nn('data/hive-version_3_prior-visual')

        forecast.model_keras_nn('data/maven-version_2_prior-visual')
        forecast.model_keras_nn('data/maven-version_3_prior-visual')

        forecast.model_keras_nn('data/lucene-version_3_prior-visual')
        forecast.model_keras_nn('data/lucene-version_4_prior-visual')
        forecast.model_keras_nn('data/lucene-version_5_prior-visual')

        forecast.model_keras_nn('data/cassandra-version_2_prior-1')
        forecast.model_keras_nn('data/cassandra-version_3_prior-1')


if __name__ == "__main__":
    main(sys.argv[1])
