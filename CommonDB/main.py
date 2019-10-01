import mysql.connector

USER = 'user'
DB = 'forecast'
PWD = 'userpwd'
HOST = 'localhost'


def connect():
    return mysql.connector.connect(host=HOST, user=USER, passwd=PWD, database=DB)


def main():
    # Database Connector
    dbc = connect()
    print(dbc)


if __name__ == "__main__":
    main()
