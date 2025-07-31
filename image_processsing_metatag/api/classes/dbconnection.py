import json
import os
import mysql.connector # type: ignore
from helper import getConfigInfo

CONTENT_USER    = "web_app"
CONTENT_PWD     = "!5@uGuST1($7FrEe1Ndi@"

DB_SERVER_LIVE_LB   = "192.168.131.29"
DB_SERVER_LIVE_39   = "192.168.12.39"
DB_SERVER_VLC_236   = '192.168.20.236'
DB_SERVER_VLC_132   = '192.168.17.132'
DB_SERVER_VLC_40    = '192.168.12.40'
DB_SERVER_VLC_DEV   = '192.168.13.90'
DB_SERVER_MEIDC     = '192.168.1.233'

DATABASE_PROD = "db_product"
DB_MEIDC      = 'online_regis1'

class DbConnection():
    
    def db_connect_dev(self):
        connect = mysql.connector.connect(
                        host     = DB_SERVER_VLC_DEV,
                        user     = CONTENT_USER,
                        password = CONTENT_PWD,
                        database = DATABASE_PROD
                    )
        # db_cursor = connect.cursor()
        return connect
    
    def db_connect_slave(self):
        connect = mysql.connector.connect(
                        host     = DB_SERVER_VLC_236,
                        user     = CONTENT_USER,
                        password = CONTENT_PWD,
                        database = DATABASE_PROD
                    )
        return connect
    
    def db_connect_live(self):
        connect = mysql.connector.connect(
                        host     = DB_SERVER_LIVE_LB,
                        user     = CONTENT_USER,
                        password = CONTENT_PWD,
                        database = DATABASE_PROD
                    )
        return connect
    
    def db_connect_meidc(self):
        connect = mysql.connector.connect(
                        host     = DB_SERVER_MEIDC,
                        user     = CONTENT_USER,
                        password = CONTENT_PWD,
                        database = DB_MEIDC
                    )
        return connect