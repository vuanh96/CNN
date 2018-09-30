import numpy as np
from scipy import sparse


def softmax(V):
    e_V = np.exp(V - np.max(V, axis=0, keepdims=True))
    Z = e_V / e_V.sum(axis=0)
    return Z


def convert_labels(y, C=3):
    Y = sparse.coo_matrix((np.ones_like(y),
                           (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y


# cost or loss function
def cost(Y, Yhat):
    return -np.sum(Y * np.log(Yhat)) / Y.shape[1]


# Generate data
N = 100  # number of points per class
d0 = 2  # dimensionality
C = 3  # number of classes
X = np.zeros((d0, N * C))  # data matrix (each row = single example)
y = np.zeros(N * C, dtype='uint8')  # class labels

for j in range(C):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[:, ix] = np.c_[r * np.sin(t), r * np.cos(t)].T
    y[ix] = j

d = [2, 100, C]  # size of layers [input, hiddens, output]
n_layers = len(d)
# initialize parameters randomly
W = [0]
b = [0]
for i in range(1, n_layers):
    W.append(0.01 * np.random.randn(d[i - 1], d[i]))
    b.append(np.zeros((d[i], 1)))

Y = convert_labels(y, C)
N = X.shape[1]
eta = 1  # learning rate
Z = [0] * n_layers
A = [0] * n_layers
for it in range(10000):
    # Feedforward
    for i in range(1, n_layers):
        if i == 1:
            Z[i] = np.dot(W[i].T, X) + b[i]
        else:
            Z[i] = np.dot(W[i].T, A[i-1]) + b[i]
        if i != n_layers - 1:
            A[i] = np.maximum(Z[i], 0)
        else:
            A[i] = softmax(Z[i])
    Yhat = A[n_layers - 1]

    # print loss after each 1000 iterations
    if it % 1000 == 0:
        # compute the loss: average cross-entropy loss
        loss = cost(Y, Yhat)
        print("iter %d, loss: %f" % (it, loss))

    # backpropagation
    E = [0] * n_layers
    dW = [0] * n_layers
    db = [0] * n_layers
    for i in range(n_layers-1, 0, -1):
        if i == n_layers - 1:
            E[i] = (Yhat - Y) / N
        else:
            E[i] = np.dot(W[i + 1], E[i + 1])
            E[i][Z[i] <= 0] = 0  # gradient of ReLU
        if i != 1:
            dW[i] = np.dot(A[i - 1], E[i].T)
        else:
            dW[i] = np.dot(X, E[i].T)
        db[i] = np.sum(E[i], axis=1, keepdims=True)

        # Gradient Descent update
        W[i] += -eta * dW[i]
        b[i] += -eta * db[i]

# # accuracy 99.33%
# for i in range(1, n_layers):
#     if i == 1:
#         Z[i] = np.dot(W[i].T, X) + b[i]
#     else:
#         Z[i] = np.dot(W[i].T, A[i - 1]) + b[i]
#     A[i]
# Z1 = np.dot(W1.T, X) + b1
# A1 = np.maximum(Z1, 0)
# Z2 = np.dot(W2.T, A1) + b2
# predicted_class = np.argmax(Z2, axis=0)
# print('training accuracy: %.2f %%' % (100 * np.mean(predicted_class == y)))
