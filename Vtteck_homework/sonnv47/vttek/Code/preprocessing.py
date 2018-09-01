import sys, os
from time import gmtime, time
import pandas as pd
import numpy as np
from multiprocessing import Pool


def csv_to_db(file_csv):
    file_name = file_csv.split('/')[-1]
    day = file_name.split('_')[0]
    database = pd.HDFStore('database.h5', 'a')
    dict_users = {}
    i = 0
    df = pd.read_csv(file_csv)
    unique_users = np.unique(df['account'].values)
    num_unique_users = len(unique_users)
    print("Number of unique users: ", num_unique_users)
    for user in unique_users:
        dict_users[user] = {}

    for index, row in df.iterrows():
        # print(row)
        arr = list(row.values)  # time, account, voice_duration, num_sms, size_data
        timer = gmtime(arr[0] / 1000.)
        user = arr[1]

        if timer.tm_hour not in dict_users[user].keys():
            dict_users[user][timer.tm_hour] = arr[2:]
        else:
            dict_users[user][timer.tm_hour] += arr[2:]
        if (index + 1) % 1000000 == 0:
            print(index, len(dict_users.keys()))

    del df
    del unique_users
    frame = []
    index = []
    t1 = time()
    for i, user in enumerate(dict_users.keys()):
        for hour in dict_users[user].keys():
            values = dict_users[user][hour]
            index.append(user)
            frame.append([hour, values[0], values[1], values[2]])
        if ((i + 1) % 100000 == 0 or (i + 1 == num_unique_users)):
            frame = pd.DataFrame(frame, index=index, columns=['hour', 'voice', 'sms', 'data'])
            t2 = time()
            database.append('day' + day, frame, data_columns=True, complevel=9, complib='blosc')
            print(time() - t2)
            index = []
            frame = []

    print(time() - t1)

    database.close()


def create_subnetwork(file_database, file_subnet, num_users):
    db = pd.HDFStore(file_database)
    list_days = list(map(lambda x: x[8:] + x[6:8] + x[4:6], db.keys()))
    list_days.sort()
    list_days = list(map(lambda x: 'day' + x[6:] + x[4:6] + x[:4], list_days))

    subnet_db = pd.HDFStore(file_subnet, 'a')
    first_day = list_days[0][1:]
    users_in_first_day = db[first_day].index.values
    users_in_first_day = np.unique(users_in_first_day)
    subnet_users = np.random.choice(users_in_first_day, size=num_users, replace=False)
    for day in list_days:
        day = day[1:]
        print(day)
        df = db[day]
        unique_users = np.unique(df.index.values)
        common_users = np.intersect1d(subnet_users, unique_users, assume_unique=True)
        df = df.loc[common_users]
        subnet_db.append(day, df, data_columns=True, complevel=0, complib='blosc')

    db.close()
    subnet_db.close()


def assign_label(file_subnet, file_activity_users, thresh_voice, thresh_sms, thresh_data, num_processor=8):
    db = pd.HDFStore(file_subnet)
    act_users = pd.HDFStore(file_activity_users, 'w')
    list_days = list(map(lambda x: x[8:] + x[6:8] + x[4:6], db.keys()))
    list_days.sort()
    list_days = list(map(lambda x: 'day' + x[6:] + x[4:6] + x[:4], list_days))

    first_day = 0
    dict_days = dict()
    print(list_days)

    for index, day in enumerate(list_days):
        day_of_month = int(day[3:5])
        month = int(day[5:7])
        year = int(day[7:])
        day_of_week = datatime.date(year, month, day_of_month)
        day_of_week = day_of_week.weekday()

        if index == 0:
            first_day = day_of_week
        week = int((first_day + index) / 7)
        dict_days[day] = {'dow': day_of_week, "dom": day_of_month, 'month': month, 'week': week}

    list_users = np.unique(db[list_days[0]].index().values)

    num_users_per_processor = int(len(list_users) / num_processor)
    list_file_saved = [
        'data/crdserver/data1/user%d-%d.csv' % (i * num_users_per_processor + 1, (i + 1) * num_users_per_processor) for
        i in range(num_processor)]

    for p in range(num_processor):
        start = p * num_users_per_processor
        end = (p + 1) * num_users_per_processor
        time_series_users = {list_users[i]: None for i in range(start, end)}
        for id_day, day in enumerate(list_days):
            t1 = time()
            df = db[day]
            dday = dict_days[day]
            unique_users_in_day = np.unique(df.index.values)
            t2 = time()

            for id_user in range(start, end):
                user = list_users[id_user]
                if user in unique_users_in_day:
                    df_user = df.loc[user]
                    if len(df_user.shape) == 1:
                        rows = np.zeros((1, 10))
                        values = df_user.values[np.newaxis, :]
                    else:
                        rows = np.zeros(df_user.shape[0], 10)
                        values = df_user.values

                    del df_user

                    rows[:, 0] = id_day
                    rows[:, 1] = values[:, 0]
                    rows[:, 6], rows[:, 7], rows[:, 8] = values[:, 1] / 60, values[:, 2], values[:, 3] / 1024 / 1024
                    rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5] = dday['dow'], dday['dom'], dday['month'], dday[
                        'week']
                    total_voice, total_sms, total_data = sum(rows[:, 6]), sum(rows[:, 7]), sum(rows[:, 8])

                    if (total_voice > thresh_voice) or (total_sms > thresh_sms) or (total_data > thresh_data):
                        rows[:, 9] = 0
                    else:
                        rows[:, 9] = 1

                    if time_series_users[user] is None:
                        time_series_users[user] = rows
                    else:
                        time_series_users[user] = np.concatenate((time_series_users, rows), axis=0)

            print('Done day %d: %s in %fs' % (id_day, day, time() - t2))

        dataframe = None
        for id_user in range(start, end):
            print(id_user)
            user = list_users[id_user]
            length = len(time_series_users[user])
            indexes = [id_user for i in range(length)]
            print('Saving user...', id_user, user)
            frame = pd.DataFrame(time_series_users[user], index=indexes,
                                 columns=['id_day', 'hour', 'dow', 'dom', 'month', 'week', 'voice', 'sms', 'data',
                                          'label'])
            if dataframe is None:
                dataframe = frame
            else:
                dataframe = dataframe.append(frame, ignore_index=False)

            if ((id_user + 1) % 1000 == 0) or (id_user == end - 1):
                if (id_user + 1) == 1000:
                    dataframe.to_csv(list_file_saved[p])
                else:
                    dataframe.to_csv(list_file_saved[p], mode='a', header=False)
                dataframe = None

        del dataframe
        print('Done save subset users %d' % p)

        del time_series_users

    db.close()


def parallel(args):
    type_cols = {'id_day': np.int32, 'hour': np.int32, 'dow': np.int32, 'dom': np.int32,
                 'week': np.int32, 'month': np.int32, 'voice': np.int32, 'sms': np.int32, 'data': np.float64,
                 'labels': np.int32}
    num_weeks, i, file = args[0], args[1], args[2]
    print(file)
    if i > 0:
        df = pd.read_csv(file, dtype=type_cols, index_col=0,
                         names=['index', 'id_day', 'hour', 'dow', 'dom', 'month', 'week', 'voice', 'sms', 'data',
                                'label'])
    else:
        df = pd.read_csv(file, dtype=type_cols, index_col=0)
    id_users = np.unique(df.index.values)
    label_week_of_users = list()

    for id_user in id_users:
        print(id_user)
        df_user = df.loc[id_user]
        if len(df_user.shape) == 1:
            df_user = df.loc[id_user]

        weeks = np.unique(df_user['week'].values)
        week_labels = np.ones(num_weeks, dtype=np.int32)
        for id_week in weeks:
            df_cur_week = df_user.loc[df_user['week'] == id_week]
            arr = np.unique(df_cur_week[['id_day', 'label']].values, axis=0)
            num_nonusing_days = 7 - len(arr)
            num_inactive_days = np.sum(arr[:, 1])
            if (num_inactive_days + num_nonusing_days > 4):
                week_labels[id_week] = 0  # inactive
        label_week_of_users.append(week_labels)
    weeks = ['week ' + str(i) for i in range(num_weeks)]
    return pd.DataFrame(np.array(label_week_of_users), index=id_users, columns=weeks)


def statistic(num_weeks, *list_file):
    list_args = list(zip([num_weeks for i in range(len(list_file))], range(len(list_file)), list(list_file)))
    p = Pool(processes=8)
    results = p.map(parallel, list_args)
    results_df = results[0]
    for i in range(1, len(results)):
        results_df = results_df.append(results[i], ignore_index=False)

    statistic = np.sum(results_df.values, axis=0)
    print('Number of active users: ', statistic)
    print('Number of inactive users: ', 500000 - statistic)

    results_df.to_csv('labels.csv')


if __name__ == '__main__':
    statistic(11, './a0.csv')
