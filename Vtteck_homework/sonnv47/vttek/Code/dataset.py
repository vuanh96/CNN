import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
import warnings
warnings.simplefilter(action='ignore', category='FutureWarning')


class Dataset:
    def __int__(self, list_file_input, file_label, shuffle=False, ignore_normalize=False):
        """
        A class for loading data for training and test model
        """
        self.list_file_input = list_file_input
        self.file_label = file_label
        self.shuffle = shuffle

        num_files = len(list_file_input)
        list_id_files = np.arange(num_files)
        if shuffle:
            np.random.shuffle(list_file_input)
        self.list_id_files = list_id_files

        self.labels = pd.read_csv(self.file_label, index_col=0)
        self.num_users = self.labels.shape[0]
        self.num_features = 24*7*3
        self.num_cols = ['index', 'id_day', 'hour', 'dow', 'dom', 'month', 'week', 'voice', 'sms', 'data', 'label']

        if not ignore_normalize:
            self.max_val_voice, self.max_val_sms, self.max_val_data = self.get_normalization_parameter()
        else:
            self.max_val_voice, self.max_val_sms, self.max_val_data = 1., 1., 1.

    def get_normalization_parameter(self):
        max_voice = 0.
        max_sms = 0.
        max_data = 0.

        for file in self.list_file_input:
            df_sub_users = pd.read_csv(file, index_col=0, names=self.num_cols, usecols=['index', 'voice', 'sms', 'data'])
            voice = max(df_sub_users['voice'].values)
            sms = max(df_sub_users['sms'].values)
            data = max(df_sub_users['data'].values)

            print(file, voice, sms, data)

            if max_voice < voice:
                max_voice = voice
            if max_sms < sms:
                max_sms = sms
            if max_data < data:
                max_data = data
        return max_voice, max_sms, max_data

    def load_minibatch(self, minibatch_size, weeks, load_all_file=False, num_epochs=None):
        generator = self.FileReader(weeks, self.list_file_input, self.max_val_voice, self.max_val_sms, self.max_val_data, minibatch_size=minibatch_size,
                                    load_all_file=load_all_file, num_epochs=num_epochs)
        return generator

    def load_all_user_at_weeks(self, weeks):
        reader = self.FileReader(weeks, self.list_file_input, self.labels, self.max_val_voice, self.max_val_sms, self.max_val_data)
        return reader.load_file(self.list_file_input)

    class FileReader:
        def __init__(self, weeks, list_file_input, labels, max_voice, max_sms, max_data, minibatch_size=None, load_all_file=False, num_epochs=none):
            self.minibatch_size = minibatch_size
            self.max_voice = max_voice
            self.max_sms = max_sms
            self.max_data = max_data
            self.weeks = weeks
            self.load_all_file = load_all_file
            self.num_epochs = num_epochs
            self.labels = labels

            if minibatch_size is not None:
                self.id_file = 0
                if load_all_file is False:
                    self.content_of_file = self.load_file(list_file_input[0])
                else:
                    print('Load all training data')
                    train_ids = np.load('train/ids.npy')
                    train_inputs = {}
                    train_labels = {}
                    for week in range(7):
                        train_inputs[week] = load_npz('./train/inputs_week'+str(week)+'.npz')
                        train_labels[week] = np.load('./train/labels_week'+str(week)+'.npy')
                    self.content_of_file = (train_ids, train_inputs, train_labels)
                    self.id_epoch = 0
                    print('Done')
                self.id_shuffled = np.arange(len(self.content_of_file[0]))
                np.random.shuffle(self.id_shuffled)
                self.id_minibatch = 0

        def load_file(self, list_files):
            num_users = 0
            map_id2id = {}
            index = 0
            for file in list_files:
                fp = open(file)
                id_start, id_end = 0, 0
                for i, line in enumerate(fp):
                    line = line.strip().split(',')
                    id_end = int(line[0])
                    if i == 0:
                        id_start = id_end
                for i in range(id_start, id_end+1):
                    map_id2id[i] = index
                    index += 1
                num_users = num_users + id_end - id_start +1
                print(num_users)

            input_every_week = {week: lil_matrix((num_users, 24*7*3)) for week in self.weeks}
            id_users = list()

            for file in list_files:
                fp = open(file)
                prev_user = None
                id_user = None
                weeks_of_user = {}
                for i,line in enumerate(fp):
                    parser = line.strip().split('.')
                    id_user = int(parser[0])
                    week = int(float(parser[6]))
                    arr = np.array([float(parser[7])/self.max_voice, float(parser[8])/self.max_sms, float(parser[9])/self.max_data])
                    if week in self.weeks:
                        if len(list(weeks_of_user.keys())) == 0:
                            weeks_of_user = {week: np.zeros((7,24,3))}
                            weeks_of_user[week][int(float(parser[3])), int(float(parser[2])),:] = arr
                        else:
                            if week in weeks_of_user.keys():
                                weeks_of_user[week][int(float(parser[3])), int(float(parser[2])), :] = arr
                            else:
                                for old_week in weeks_of_user.keys():
                                    input_every_week[old_week][map_id2id[prev_user], :] = weeks_of_user[old_week].flatten()
                                weeks_of_user = {week: np.zeros((7,24,3))}
                                weeks_of_user[week][int(float(parser[3])), int(float(parser[2])), :] = arr

                    else:
                        if len(list(weeks_of_user.keys())) > 0:
                            for old_week in weeks_of_user.keys():
                                input_every_week[old_week][map_id2id[prev_user], :] = weeks_of_user[old_week].flatten()
                            weeks_of_user = {}

                    if (prev_user is not None) and (prev_user != id_user):
                        id_users.append(prev_user)
                    prev_user = id_user
                for week in weeks_of_user.keys():
                    input_every_week[week][map_id2id[prev_user], :] = weeks_of_user[week].flatten()
                id_users.append(prev_user)

                for week in input_every_week.keys():
                    input_every_week[week] = input_every_week.tocsr()

                labels_of_users = self.labels.loc[id_users]
                labels_of_users = {week: labels_of_users['week_'+str(week+1)].values for week in self.weeks}
                return np.array(id_users), input_every_week, labels_of_users

            def __iter__(self):
                return self

            def __next__(self):
                return self.next()

            def next(self):
                id_users, subset_users, subset_labels = self.content_of_file
                num_users = len(id_users)
                if self.load_all_file is False:
                    stop = False
                    if self.id_minibatch >= num_users:
                        self.id_flie += 1
                        if self.id_file == len(self.list_file_input):
                            stop = True
                        else:
                            self.content_of_file = self.load_file([self.list_flie_input[self.id_file]])
                            id_users, subset_users, subset_labels = self.content_of_file
                            num_users = len(id_users)
                            self.id_minibatch = 0
                            self.id_shuffled = np.arange(num_users)
                            np.random.shuffle(self.id_shuffled)

                    if stop is not True:
                        if (self.id_minibatch + self.minibatch_size) >= num_users:
                            ids = self.id_shuffled[self.id_minibatch:]
                        else:
                            ids = self.id_shuffled[self.id_minibatch:(self.id_minibatch + self.minibatch_size)]
                        minibatch_ids = id_users[ids]
                        minibatch_inputs = {week: subset_users[week][ids].tocsr() for week in subset_users.keys()}
                        minibatch_labels = {week: subset_labels[week][ids] for week in subset_labels.keys()}
                        self.id_minibatch += self.minibatch_size
                        return minibatch_ids, minibatch_inputs, minibatch_labels
                    else:
                        raise StopIteration()
                else:
                    stop = False
                    if self.id_minibatch >= num_users:
                        self.id_epoch += 1
                        if self.id_epoch == self.num_epochs:
                            stop = True
                        else:
                            print('Epochs ', self.id_epoch+1)
                            self.id_minibatch = 0
                            self.id_shuffled = np.arange(num_users)
                            np.random.shuffle(self.id_shuffled)
                    if stop is not True:
                        if (self.id_minibatch+self.minibatch_size) >= num_users:
                            ids = self.id_shuffled[self.id_minibatch:]
                        else:
                            ids = self.id_shuffled[self.id_minibatch:(self.id_minibatch+self.minibatch_size)]
                        minibatch_ids = id_users[ids]
                        minibatch_inputs = {week: subset_users[week][ids].tocsr() for week in subset_users.keys()}
                        minibatch_labels = {week: subset_labels[week][ids] for week in subset_labels.keys()}
                        self.id_minibatch += self.minibatch_size
                        return minibatch_ids, minibatch_inputs, minibatch_labels
                    else:
                        raise StopIteration()

if __name__ == '__main__':
    list_file_input = []
    dataset =  Dataset(list_file_input, 'labels.csv', shuffle= True)
    ids, inputs, labels = dataset.load_all_user_at_weeks([7,8])
    print(ids.shape, inputs[7].shape, inputs[8].shape)





