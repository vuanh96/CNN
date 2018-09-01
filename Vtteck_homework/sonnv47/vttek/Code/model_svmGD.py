import numpy as np
import sys
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
from sklearn.metrics import classification_report, precision_recall_fscore_support, fbeta_score
from dataset import Dataset

C = 1E6
lamda = 1./C
def compute_loss(X, y, w, b):
    score = X.dot(w) + b  # shape (N,)
    y_score = y*score
    return (np.sum(np.maximum(0, 1 - y_score)) + .5*lamda*w.dot(w)) / X.shape[0]

def compute_gradient(X, y, w, b):
    score = X.dot(w) + b  # shape (N,)
    y_score = y*score  # element wise product, shape (N,)
    active_set = np.where(y_score <= 1)[0]  # consider 1 - yz >= 0 only
    temp = -X.multiply(y[:, np.newaxis])  # each row is y_n*x_n
    grad_w = (np.sum(temp.toarray()[active_set], axis=0) + lamda*w) / X.shape[0]
    grad_b = (-np.sum(y[active_set])) / X.shape[0]
    return (grad_w, grad_b)

def num_gradient(X, y, w, b):
    eps = 1e-10
    gw = np.zeros_like(w)
    for i in range(len(w)):
        wp = w.copy()
        wm = w.copy()
        wp[i] += eps
        wm[i] -= eps
        gw[i] = (compute_loss(X, y, wp, b) - compute_loss(X, y, wm, b)) / (2*eps)
    gb = (compute_loss(X, y, w, b + eps) - compute_loss(X, y, w, b - eps)) / (2*eps)
    return (gw, gb)

def check_gradient(X, y, w, b):
    w = .1*np.random.randn(X.shape[1])
    b = np.random.randn()
    (gw0, gb0) = compute_gradient(X, y, w, b)
    (gw1, gb1) = num_gradient(X, y, w, b)
    print('grad_w difference = ', np.linalg.norm(gw0 - gw1))
    print('grad_b difference = ', np.linalg.norm(gb0 - gb1))

def softmarginSVM_gd(X, y, w0, b0, eta, gamma):
    w = [w0]
    v = [np.zeros_like(w0)]  # momentum w
    b = [b0]
    c = [np.zeros_like(b0)]  # momentum b
    # check_gradient(X, y, w, b)
    for it in range(100):
        (grad_w, grad_b) = compute_gradient(X, y, w[-1]-gamma*v[-1], b[-1]-gamma*c[-1])

        v_new = gamma*v[-1] + eta*grad_w
        w_new = w[-1] - v_new
        w.append(w_new)
        v.append(v_new)
        c_new = gamma*c[-1] + eta*grad_b
        b_new = b[-1] - c_new
        b.append(b_new)
        c.append(c_new)

        if (it % 10) == 0:
            print('  iteration %d' % it + ' loss: %f' % compute_loss(X, y, w[-1], b[-1]))
    return (w[-1], b[-1])

def learn_model(dataset, train_weeks, test_weeks, minibatch_size, num_epochs, eta, gamma, load_all_file=False):

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

    w_init = .1 * np.random.randn(num_features)
    b_init = .1 * np.random.randn()
    w_hinge = [w_init]
    b_hinge = [b_init]
    num_minibatch = int(np.ceil(all_train_inputs.shape[0] / float(minibatch_size)))

    for epoch in range(num_epochs):
        print('Epoch ', epoch)
        mix_ids = np.random.permutation(all_train_inputs.shape[0])
        for it in range(num_minibatch):
            print('Minibatch ', it)
            ids = mix_ids[minibatch_size*it:min(minibatch_size*(it+1), all_test_inputs.shape[0])]
            X_batch = all_train_inputs[ids]
            y_batch = all_train_labels[ids]
            (w_new, b_new) = softmarginSVM_gd(X_batch, y_batch, w_hinge[-1], b_hinge[-1], eta, gamma)
            w_hinge.append(w_new)
            b_hinge.append(b_new)
        model = (w_hinge[-1], b_hinge[-1])
        thresh = find_boundary(model, all_train_inputs, all_train_labels)
        print('Predicted result after epoch: ', epoch)
        predict(model, all_test_inputs, all_test_labels, thresh)

def find_boundary(model, all_train_inputs, all_train_labels):
    w, b = model
    temp = all_train_inputs.dot(w) + b
    temp = temp.flatten()
    prob = 1 / (1 + np.exp(-temp))

    id_c0 = np.where(all_train_labels == -1)[0]  #inactive
    id_c1 = np.where(all_train_labels == 1)[0]  #active
    hist_c0, bins = np.histogram(prob[id_c0], bins=np.arange(0, 1.01, 0.01))
    hist_c1, bins = np.histogram(prob[id_c1], bins=np.arange(0, 1.01, 0.01))

    # print(hist_c0)
    # print(hist_c1)

    min_bin = (hist_c1>0).argmax()
    max_bin = len(hist_c0) - 1 - (hist_c0>0)[::-1].argmax()
    thresh = 0
    mis_classify = 1e9
    if min_bin < max_bin:
        for i in range(min_bin, max_bin):
            value = sum(hist_c0[i:]) + sum(hist_c1[:(i+1)])
            if value < mis_classify:
                mis_classify = value
                thresh = i
    # else:
    #     thresh = int((min_bin + max_bin)/2)
    print('min_bin: %d, max_bin: %d, mis_classify: %d' %(min_bin, max_bin, mis_classify))
    thresh = (bins[thresh] + bins[thresh+1])/2
    return thresh



def predict(model, all_test_inputs, all_test_labels, thresh):
    w_final, b_final = model[0], model[1]
    temp = all_test_inputs.dot(w_final) + b_final
    prob = 1/(1+np.exp(-temp))
    pred_labels = np.where(prob >= thresh, 1, 0)
    print(classification_report(all_test_labels, pred_labels, target_names=['inactive', 'active']))


if __name__ == '__main__':
    load_all_file = sys.argv[1]
    list_file_input=[]
    dataset = Dataset(list_file_input, 'label.csv', shuffle=True, ignore_normalize=True)
    learn_model(dataset, [0,1,2,3,4,5,6], [7,8], minibatch_size=70000, num_epochs=100, eta=0.09, gamma=0.1, load_all_file=load_all_file)

