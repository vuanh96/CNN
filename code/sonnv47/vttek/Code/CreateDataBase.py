import sqlite3
import numpy as np
import pandas as pd
from sqlite3 import Error

data = pd.DataFrame(columns=['time', 'account', 'voice_duration', 'num_sms', 'size_data'])


def create_connection(database_file):
    try:
        conn = sqlite3.connect(database_file)
        return conn
    except Error as e:
        print(e)

    return None


def select_all_task(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM cdr06042018 LIMIT 100, 2000")
    rows = cur.fetchall()
    for row in rows:
        data.add(row)
        print(data)


def main():
    database = "/home/..."
    conn = create_connection(database)
    with conn:
        select_all_task(conn)


if __name__ == '__main__':
    main()
