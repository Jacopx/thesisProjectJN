import sys
import general
import distance
import duration
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
        # forecast.random_forest(dbc, 'data/DateCount_21_BS')
        # forecast.random_forest(dbc, 'data/DateCount_23_BS')
        # forecast.random_forest(dbc, 'data/DateCount_24_BS')
        # forecast.random_forest(dbc, 'data/DateCount_28_BS')
        # forecast.random_forest(dbc, 'data/DateCount_46_BS')
        # forecast.random_forest(dbc, 'data/DateCount_50_BS')
        # forecast.random_forest(dbc, 'data/DateCount_59_BS')
        # forecast.random_forest(dbc, 'data/DateCount_60_BS')
        # forecast.random_forest(dbc, 'data/DateCount_69_BS')
        # forecast.random_forest(dbc, 'data/DateCount_70_BS')
        # forecast.random_forest(dbc, 'data/DateCount_BS')
        # forecast.random_forest(dbc, 'data/DateHourMinuteCount_70_BS')
        forecast.random_forest(dbc, 'data/station70')
    else:
        forecast.random_forest(dbc, 'data/satur')


if __name__ == "__main__":
    main(sys.argv[1])
