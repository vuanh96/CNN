import numpy as np
import time
import sys
from scipy.sparse import csr_matrix, lil_matrix, load_npz
from sklearn.metrics import classification_report, precision_recall_fscore_support, fbeta_score
from sklearn.svm import SVC, LinearSVC
from dataset import Dataset

def learn_model(dataset, train_weeks, test_weeks, load_all_file=False):

    num_users = dataset.num_users
    num_features = dataset.num_features
    num_train_week = len(train_weeks)
    num_test_week = len(test_weeks)

    print('Load training data...')
    train_ids = np.load('/home/haitm121/Desktop/train/ids.npy')
    train_inputs = {}
    train_labels = {}
    for week in train_weeks:
        train_inputs[week] = load_npz('/home/haitm121/Desktop/train/inputs_week'+str(week)+'.npz')
        train_labels[week] = np.load('/home/haitm121/Desktop/train/labels_week'+str(week)+'.npy')
    all_train_inputs = lil_matrix((num_users*num_train_week, num_features))
    all_train_labels = np.random.rand(num_users*num_train_week)
    for i, week in enumerate(train_weeks):
        print('week: ', week)
        all_train_inputs[i*num_users:(i+1)*num_users, :] = train_inputs[week]
        # del train_inputs[week]
        all_train_labels[i*num_users:(i+1)*num_users] = train_labels[week]
        # del train_labels[week]
    print('Done!')

    print('Load test data...')
    test_ids = np.load('/home/haitm121/Desktop/test/ids.npy')
    test_inputs = {}
    test_labels = {}
    for week in test_weeks:
        test_inputs[week] = load_npz('/home/haitm121/Desktop/test/inputs_week'+str(week)+'.npz')
        test_labels[week] = np.load('/home/haitm121/Desktop/test/labels_week'+str(week)+'.npy')
    all_test_inputs = lil_matrix((num_users*num_test_week, num_features))
    all_test_labels = np.random.rand(num_users*num_test_week)
    for i, week in enumerate(test_weeks):
        print('week: ', week)
        all_test_inputs[i*num_users:(i+1)*num_users, :] = test_inputs[week]
        all_test_labels[i*num_users:(i+1)*num_users] = test_labels[week]
    print('Done!')

    print('Training model...')
    t1 = time.time()
    clf = LinearSVC(C=1e6, verbose=True, max_iter=10000)
    # clf = SVC(kernel='rbf', C=1e6, verbose=True, max_iter=10000)

    clf.fit(all_test_inputs.tocsr(), all_train_labels)
    t2 = time.time()
    print('Done!')
    print('Total time training: ', t2-t1)

    models = (clf.coef_.flatten(), clf.intercept_)
    pred_labels = clf.predict(all_test_inputs.tocsr())
    print('Prediction result of LinearSVC model:')
    # print('Prediction result of GaussianSVC model:')
    print(classification_report(all_test_labels, pred_labels, target_names=['inactive', 'active']))

def libsvm(all_train_inputs, all_train_labels):
    nonzero_data = all_train_inputs.nonzero()
    temp_file = []
    num_samples = all_train_inputs.shape[0]
    for i in range(num_samples):
        dict_row = {}
        temp = np.where(nonzero_data[0] == i)[0]
        for j in nonzero_data[1][temp]:
            dict_row[j+1] = all_train_inputs[i, j]
        temp_file.append(dict_row)

    file_inputs = open('file_input_svm.txt', 'w')
    for idx, row in enumerate(temp_file):
        file_inputs.write('%d ' % all_train_labels[idx])
        row_convert = list(row.keys())
        for key in row_convert[:-1]:
            file_inputs.write('%d:%d ' % (key, row[key]))
        file_inputs.write('%d %d:%d' % (all_train_labels[num_samples-1], row_convert[-1], row[row_convert[-1]]))
        file_inputs.write('\n')
    file_inputs.close()


if __name__ == '__main__':
    load_all_file = sys.argv[1]
    list_file_input=[]
    dataset = Dataset(list_file_input, 'label.csv', shuffle=True, ignore_normalize=True)
    learn_model(dataset, [0,1,2,3,4,5,6], [7,8], C=1e6, load_all_file=load_all_file)
