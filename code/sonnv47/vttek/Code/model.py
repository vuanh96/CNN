import sys
import numpy as np
from dataset import Dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support, fbeta_score
from scipy.sparse import save_npz, load_npz


def compute_loss_and_gradient(inputs, labels, w_global, w_personal, intercept, alpha, beta, lambda_global,
                              lambda_personal, class_weights=None, C=1e6):
    if class_weights is None:
        c0, c1 = 1, 1
    else:
        c0, c1 = class_weights[0], class_weights[1]

    num_weeks = len(list(inputs.keys()))
    num_users, num_features = list(inputs.values())[0].shape
    temp = np.zeros((num_users, num_features))
    matrix_labels = np.zeros((num_users, num_features))

    weeks = list(inputs.keys())
    weeks.sort()

    for i, week in enumerate(weeks):
        matrix_labels[:, i] = 2 * labels[weeks] - 1
        mul = inputs[week].multiply(w_global + w_personal)
        mul = np.array(mul.sum(axis=1)) + intercept
        mul = matrix_labels[:, i] * mul.flatten()
        temp[:, i] = mul

        # apply sigmoid function
        # sigmoid = 1/(1+np.exp(-beta*temp))
        # matrix_loss = beta*c0*(1-matrix_labels)*temp - ((c1-c0)*matrix_labels + c0)*np.log(sigmoid)
        matrix_loss = np.maximum(0, 1 - temp)

        decay_vector = np.array([np.exp(alpha * (t - num_weeks)) for t in range(1, num_weeks + 1)])
        regular_term = lambda_global * np.sum(w_global ** 2) / 2 + lambda_personal * np.sum(w_personal ** 2) / 2

        total_loss = (np.sum(decay_vector * matrix_loss) + regular_term) / num_users

        active_set = np.where(1 - temp > 0, 1, 0)

        # matrix_grad = (sigmoid*((c1-c0)*matrix_labels + c0) - c1*matrix_labels)*decay_vector
        matrix_grad = -active_set * matrix_labels * decay_vector
        grad_w_global = 0
        grad_w_personal = 0
        grad_intercept = 0

        for i, week in enumerate(weeks):
            grad_w_global += inputs[week].transpose().dot(matrix_grad[:, i])
            grad_w_personal += inputs[week].toarray() + matrix_grad[:, i][:, np.newaxis]
        grad_w_global = (grad_w_global + lambda_global * w_global) / num_users
        grad_w_personal = (grad_w_personal + lambda_personal * w_personal) / num_users
        grad_intercept = np.sum(matrix_grad) / num_users

        # grad_w_global = (np.sum(matrix_grad, axis=(0,1)) + lambda_global * w_global) / num_users
        # grad_w_personal = (np.sum(matrix_grad, axis=1) + lambda_personal * w_personal) / num_users

        return total_loss, grad_w_global, grad_w_personal, grad_intercept


def learn_model(dataset, alpha, beta, lambda_global, lambda_personal, training_weeks, test_weeks, num_epochs=100,
                minibatch_size=10000, tau=100.0, kappa=0.9,
                load_all_file=False, class_weights=None, C=1e6):
    num_users = dataset.num_users
    num_features = dataset.num_features

    W_global = np.random.rand(num_features)
    W_personal = np.random.rand(num_users, num_features)
    W_global /= np.sum(W_global)
    W_personal /= np.sum(W_personal, axis=1)[:, np.newaxis]
    intercept = np.random.rand()

    print('Load training data at week %d... ' % training_weeks[-1])

    ids = np.load('/home/haitm121/Desktop/week6/ids.npy')
    inputs = {}
    labels = {}
    inputs[6] = load_npz('/home/haitm121/Desktop/week6/inputs.npz')
    labels[6] = np.load('/home/haitm121/Desktop/week6/labels.npy')
    print('...Done')
    data_last_training_week = (ids, inputs, labels)

    print('Load test data...')
    test_ids = np.load('/home/haitm121/Desktop/test/ids.npy')
    test_inputs = {}
    test_labels = {}
    for week in test_weeks:
        test_inputs[week] = load_npz('/home/haitm121/Desktop/test/inputs_week' + str(week) + '.npz')
        test_labels[week] = np.load('/home/haitm121/Desktop/test/labels_week' + str(week) + '.npy')
    print('...Done')
    test_data = (test_ids, test_inputs, test_labels)

    num_iters = num_epochs
    if load_all_file is True:
        num_iters = 1
    t = 1
    for i in range(num_iters):
        if load_all_file is not True:
            print('Epoch %d: ' % (i + 1))
        minibatches = dataset.load_minibatch(minibatch_size, training_weeks, load_all_file=load_all_file,
                                             num_epochs=num_epochs)
        for minibatch in minibatches:
            minibatch_ids, minibatch_inputs, minibatch_labels = minibatch[0], minibatch[1], minibatch[2]

            total_loss, grad_w_global, grad_w_personal, grad_intercept = compute_loss_and_gradient(minibatch_inputs,
                                                                                                   minibatch_labels,
                                                                                                   W_global, W_personal[
                                                                                                       minibatch_ids],
                                                                                                   intercept, alpha,
                                                                                                   beta, lambda_global,
                                                                                                   lambda_personal,
                                                                                                   class_weights=class_weights,
                                                                                                   C=C)

            learning_rate = np.power(t + tau, -kappa)

            W_global = W_global - learning_rate * grad_w_global
            W_personal[minibatch_ids] = W_personal[minibatch_ids] - learning_rate * grad_w_personal
            intercept = intercept - learning_rate * grad_intercept

            print('Minibatch %d, loss %f' % (t, total_loss))
            if t % 10 == 0:
                models = (W_global, W_personal, intercept)
                thresh = find_boundary(models, data_last_training_week, training_weeks[-1], beta)
                print('Thresh: ', thresh)
                predict(models, test_data, thresh, beta)

    np.save('/home/haitm121/Desktop/save/w0.npy', W_global)
    np.save('/home/haitm121/Desktop/save/wi.npy', W_personal)
    np.save('/home/haitm121/Desktop/save/intercept.npy', intercept)


def find_boundary(models, training_data, week, beta):
    w_global, w_personal, intercept = models
    id_users, inputs, labels = training_data
    prob = inputs[week].multiply(w_global + w_personal)
    prob = np.array(prob.sum(axis=1)) + intercept
    prob = prob.flatten()

    true_labels = labels[week]

    id_c0 = np.where(true_labels == 0)[0]  # inactive
    id_c1 = np.where(true_labels == 1)[0]  # active
    hist_c0, bins = np.histogram(prob[id_c0], bins=np.arange(0, 1.01, 0.01))
    hist_c1, bins = np.histogram(prob[id_c1], bins=np.arange(0, 1.01, 0.01))

    # print(hist_c0)
    # print(hist_c1)

    min_bin = (hist_c1 > 0).argmax()
    max_bin = len(hist_c0) - 1 - (hist_c0 > 0)[::-1].argmax()
    thresh = 0
    mis_classify = 1e9
    if min_bin < max_bin:
        for i in range(min_bin, max_bin):
            value = sum(hist_c0[i:]) + sum(hist_c1[:(i + 1)])
            if value < mis_classify:
                mis_classify = value
                thresh = i
    # else:
    #     thresh = int((min_bin + max_bin)/2)
    print('min_bin: %d, max_bin: %d, mis_classify: %d' % (min_bin, max_bin, mis_classify))
    thresh = (bins[thresh] + bins[thresh + 1]) / 2
    return thresh


def predict(models, test_data, thresh, beta):
    w_global, w_personal, intercept = models
    id_users, inputs, labels = test_data

    for week in inputs.keys():
        prob = inputs[week].multiply(w_global + w_personal)
        prob = np.array(prob.sum(axis=1)) + intercept
        prob = prob.flatten()

        prob = 1 / (1 + np.exp(-beta * prob))
        print('Probability at week ', week, prob)

        pred_labels = np.array(prob > thresh, dtype=np.int32)
        true_labels = labels[week]
        print(classification_report(true_labels, pred_labels, target_names=['inactive', 'active']))


if __name__ == '__main__':
    load_all_file = sys.argv[1]
    list_file_input = []
    dataset = Dataset(list_file_input, 'label.csv', shuffle=True, ignore_normalize=True)
    learn_model(dataset, 0.1, 1, 1, 1, [0, 1, 2, 3, 4, 5, 6], [7, 8], num_epochs=100, minibatch_size=10000, tau=100.0,
                kappa=0.9,
                load_all_file=True, class_weights=[1.0, 1.0], C=1)
