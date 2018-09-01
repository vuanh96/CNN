from __future__ import division, print_function
import numpy as np
from random import choice
import sys
from scipy.sparse import csr_matrix, lil_matrix, load_npz
from sklearn.metrics import classification_report, precision_recall_fscore_support, fbeta_score
from dataset import Dataset

class SVM_SMO():
    def __init__(self, max_iter=10000, kernel_type='gaussian', C=1.0, epsilon=0.001):
        self.kernels = {
            'gaussian': self.kernel_gaussian
        }
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.C = C
        self.epsilon = epsilon

    def fit(self, X_train, Y_train, X_test, Y_test):
        # initialization
        num_users, num_features = X_train.shape[0], X_train.shape[1]
        # num_users, num_features = X_train.shape
        alpha = self.C * np.random.rand(num_users)
        kernel = self.kernels[self.kernel_type]
        count = 0

        while True:
            count += 1
            print('Iteraction ', count)
            alpha_prev = np.copy(alpha)
            for j in range(0, num_users):
                i = choice([i for i in range(0, num_users) if i not in [j]])
                x_i, x_j, y_i, y_j = X_train[i,:].toarray().flatten(), X_train[j,:].toarray().flatten(), Y_train[i], Y_train[j]
                k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2*kernel(x_i, x_j)
                if k_ij == 0:
                    continue

                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                (L, H) = self.compute_low_high(self.C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute E_i, E_j
                error_i = self.error(x_i, y_i, X_train, Y_train, alpha)
                error_j = self.error(x_j, y_j, X_train, Y_train, alpha)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j * (error_i - error_j)) / k_ij
                alpha[i] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)

                alpha[i] = alpha_prime_i + y_i*y_j*(alpha_prime_j - alpha[j])

            if count % 10 == 0:
                self.evaluate(X_train, Y_train, X_test, Y_test, alpha)
            # Check convergence
            diff = np.linalg.norm(alpha-alpha_prev)
            if diff < self.epsilon:
                break
            if count >= self.max_iter:
                print('Iteration number exceeded the max of %d iterations' % (self.max_iter))

        # Get support vectors
        alpha_idx = np.where(alpha > 1e-5)[0]
        support_vectors = X_train[alpha_idx, :]

        return support_vectors, alpha

    def compute_score(self, x, X_train, Y_train, alpha):
        alpha_id0 = np.where(alpha > 1e-5)[0]
        alpha_idc = np.where((alpha > 1e-5) & (alpha < .999*self.C))[0]
        score = 0
        for index in alpha_id0:
            score += self.kernel_gaussian(X_train[index].toarray().flatten(), x)*alpha[index]*Y_train[index]
        # score = np.sum(self.kernel_gaussian(X_train[alpha_id0].toarray(), x)*alpha[alpha_id0]*Y_train[alpha_id0])
        intercept = 0
        for index_c in alpha_idc:
            intercept += Y_train[index_c]
            for index_0 in alpha_id0:
                intercept -= self.kernel_gaussian(X_train[index_0].toarray().flatten(), X_train[index_c].toarray().flatten())*alpha[index_0]*Y_train[index_0]
        # intercept = np.sum(y[alpha_idc])

        return score + intercept / len(alpha_idc)


    def evaluate(self, X_train, Y_train, X_test, Y_test, alpha):
        test_size = X_test.shape[0]
        Y_pred = np.zeros(test_size)
        for i in range(test_size):
            x, y_true = X_test[i].toarray().flatten(), Y_test[i]
            score = self.compute_score(x, X_train, Y_train, alpha)
            if score >= 0:
                Y_pred[i] = 1
        print(classification_report(Y_test, Y_pred, target_names=['inactive', 'active']))

    # Error prediction
    def error(self, x, y, X_train, Y_train, alpha):
        predict = self.compute_score(x, X_train, Y_train, alpha)
        return 2*(int(predict > 0)) - 1 - y

    def compute_low_high(self, C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if (y_i != y_j):
            return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
        else:
            return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

    def kernel_gaussian(self, x1, x2, sigma=1):
        return np.exp(-np.linalg.norm(x1 - x2) / (2*sigma**2))

def learn_model(dataset, train_weeks, test_weeks, C, load_all_file=False):

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

    all_train_labels = 2*all_train_labels-1  # convert labels to 1, -1 using for training

    svm = SVM_SMO(C=1e6)
    svm.fit(all_train_inputs, all_train_labels, all_test_inputs, all_test_labels)

if __name__ == '__main__':
    load_all_file = sys.argv[1]
    list_file_input=[]
    dataset = Dataset(list_file_input, 'label.csv', shuffle=True, ignore_normalize=True)
    learn_model(dataset, [0, 1, 2, 3, 4, 5, 6], [7, 8], C=1e6, load_all_file=load_all_file)



