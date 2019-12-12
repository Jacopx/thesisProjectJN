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
        # forecast.model_randomforest('data/hadoop_Mixed_prior-1')
        # forecast.model_keras_nn('data/hadoop_Mixed_prior-1')
        # forecast.model_keras_lstm('data/hadoop_Mixed_count-1')

        # forecast.model_ludwig('data/hadoop_prior1')

        forecast.model_keras_nn('data/hadoop_version_2_prior-1')

        # forecast.model_keras_nn('data/hadoop_Mixed_prior-1')
        # forecast.model_keras_nn('data/hadoop_Mixed_prior-2')
        # forecast.model_keras_nn('data/hadoop_Critical_prior-2')
        # forecast.model_keras_nn('data/hadoop_Major_prior-2')
        # forecast.model_keras_nn('data/hadoop_Blocker_prior-2')
        # forecast.model_keras_nn('data/hadoop_Minor_prior-2')
        # forecast.model_keras_nn('data/hadoop_Trivial_prior-2')


if __name__ == "__main__":
    main(sys.argv[1])
