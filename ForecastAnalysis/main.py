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
        # forecast.model_randomforest('data/hadoop_mixed_prior1')
        # forecast.count_model('data/hadoop_mixed_count1')
        # forecast.count_model_keras_nn('data/hadoop_mixed_prior1')
        # forecast.count_model_keras_lstm('data/hadoop_mixed_count1')

        # forecast.model_randomforest('data/hadoop_mixed_prior1')
        # forecast.model_ludwig('data/hadoop_prior1')
        forecast.model_keras_nn('data/hadoop_mixed_prior-1')
        forecast.model_keras_nn('data/hadoop_mixed_prior-2')
        forecast.model_keras_nn('data/hadoop_mixed_prior-4')
        forecast.model_keras_nn('data/hadoop_mixed_prior-6')
        forecast.model_keras_nn('data/hadoop_mixed_prior-8')
        forecast.model_keras_nn('data/hadoop_mixed_prior-10')
        forecast.model_keras_nn('data/hadoop_mixed_prior-12')
        forecast.model_keras_nn('data/hadoop_mixed_prior-16')
        forecast.model_keras_nn('data/hadoop_mixed_prior-20')
        forecast.model_keras_nn('data/hadoop_mixed_prior-30')
        forecast.model_keras_nn('data/hadoop_mixed_prior-40')
        forecast.model_keras_nn('data/hadoop_mixed_prior-52')

        # forecast.count_model_keras_nn('data/hbase_mixed_prior1')
        # forecast.count_model_keras_nn('data/hbase_mixed_prior2')
        # forecast.count_model_keras_nn('data/hbase_mixed_prior4')
        # forecast.count_model_keras_nn('data/hbase_mixed_prior6')
        # forecast.count_model_keras_nn('data/hbase_mixed_prior8')
        # forecast.count_model_keras_nn('data/hbase_mixed_prior10')
        # forecast.count_model_keras_nn('data/hbase_mixed_prior12')
        # forecast.count_model_keras_nn('data/hbase_mixed_prior16')
        #
        # forecast.count_model_keras_nn('data/hive_mixed_prior1')
        # forecast.count_model_keras_nn('data/hive_mixed_prior2')
        # forecast.count_model_keras_nn('data/hive_mixed_prior4')
        # forecast.count_model_keras_nn('data/hive_mixed_prior6')
        # forecast.count_model_keras_nn('data/hive_mixed_prior8')
        # forecast.count_model_keras_nn('data/hive_mixed_prior10')
        # forecast.count_model_keras_nn('data/hive_mixed_prior12')
        # forecast.count_model_keras_nn('data/hive_mixed_prior16')
        #
        # forecast.count_model_keras_nn('data/cassandra_mixed_prior1')
        # forecast.count_model_keras_nn('data/cassandra_mixed_prior2')
        # forecast.count_model_keras_nn('data/cassandra_mixed_prior4')
        # forecast.count_model_keras_nn('data/cassandra_mixed_prior6')
        # forecast.count_model_keras_nn('data/cassandra_mixed_prior8')
        # forecast.count_model_keras_nn('data/cassandra_mixed_prior10')
        # forecast.count_model_keras_nn('data/cassandra_mixed_prior12')
        # forecast.count_model_keras_nn('data/cassandra_mixed_prior16')
        #
        # forecast.count_model_keras_nn('data/lucene_mixed_prior1')
        # forecast.count_model_keras_nn('data/lucene_mixed_prior2')
        # forecast.count_model_keras_nn('data/lucene_mixed_prior4')
        # forecast.count_model_keras_nn('data/lucene_mixed_prior6')
        # forecast.count_model_keras_nn('data/lucene_mixed_prior8')
        # forecast.count_model_keras_nn('data/lucene_mixed_prior10')
        # forecast.count_model_keras_nn('data/lucene_mixed_prior12')
        # forecast.count_model_keras_nn('data/lucene_mixed_prior16')
        #
        # forecast.count_model_keras_nn('data/maven_mixed_prior1')
        # forecast.count_model_keras_nn('data/maven_mixed_prior2')
        # forecast.count_model_keras_nn('data/maven_mixed_prior4')
        # forecast.count_model_keras_nn('data/maven_mixed_prior6')
        # forecast.count_model_keras_nn('data/maven_mixed_prior8')
        # forecast.count_model_keras_nn('data/maven_mixed_prior12')
        # forecast.count_model_keras_nn('data/maven_mixed_prior16')

if __name__ == "__main__":
    main(sys.argv[1])
