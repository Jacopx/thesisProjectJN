import mysql.connector
import time
import sys

# FreeNAS
USER = 'eis'
DB = 'forecast'
PWD = 'eisworld2019'
HOST = 'db.jacopx.me'
PORT = '33306'


def connect():
    return mysql.connector.connect(host=HOST, port=PORT, user=USER, passwd=PWD, database=DB)


def main():
    # Database Connector
    dbc = connect()
    print(dbc)


if __name__ == "__main__":
    main()
