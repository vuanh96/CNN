import numpy as np
from scipy import sparse
import csv
import plotly as plt
import plotly.graph_objs as go


def softmax(Z):
    e_Z = np.exp((Z - np.max(Z, axis=0, keepdims=True)))
    Z = e_Z / e_Z.sum(axis=0)
    return Z


Z = [2, 3, 4, 5]


# One-hot coding
def one_hot_coding(y, C):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y


# 1. Loss function
def cost(Y, Ymu):
    return -np.sum(Y * np.log(Ymu)) / Y.shape[1]


def train(X, y, C, hidden_layer_size, num_iters):
    print("The model is training .......!!!!")
    loss = list()
    d0 = 2  # the dimension number of X
    d1 = hidden_layer_size
    d2 = C

    # Initialize parametters randomly
    W1 = 0.001 * np.random.randn(d0, d1)
    b1 = np.zeros((d1, 1))
    W2 = 0.001 * np.random.randn(d1, d2)
    b2 = np.zeros((d2, 1))

    Y = one_hot_coding(y, C)
    N = X.shape[1]  # The number of samples in train dataset
    eta = 1  # learning rate

    for i in range(num_iters):
        # 1. Feedforward to compute loss and the loss is computed here (followed the fomula)
        Z1 = np.dot(W1.T, X) + b1
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(W2.T, A1) + b2
        Ymu = softmax(Z2)
        loss.append(cost(Y, Ymu))

        # 2. Backpropagation Step
        E2 = (Ymu - Y) / N
        dW2 = np.dot(A1, E2.T)
        db2 = np.sum(E2, axis=1, keepdims=True)

        E1 = np.dot(W2, E2)
        # use gradient of RelU
        E1[Z1 <= 0] = 0
        dW1 = np.dot(X, E1.T)
        db1 = np.sum(E1, axis=1, keepdims=True)

        # Update Gradient Descent
        W1 = W1 - eta * dW1
        b1 = b1 - eta * db1
        W2 = W2 - eta * dW2
        b2 = b2 - eta * db2

    loss = np.asarray(loss)
    print("Finish training step.")
    return W1, b1, W2, b2, y, loss


def predict(X, W1, b1, W2, b2, y):
    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2

    predicted_y = np.argmax(Z2, axis=0)
    print('Training accuracy: %.2f %%' % (100 * np.mean(predicted_y == y)))
    return predicted_y


def read_data(path):
    dim_x = 2
    y = []
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i > 0:
                row[2] = float(row[2])
                y.append(row[2])

    num_samples = len(y)
    X = np.zeros((dim_x, num_samples))
    y = np.asarray(y)
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i > 0:
                X[0][i - 1] = float(row[0])
                X[1][i - 1] = float(row[1])

    return X, y


def draw_loss(arr1, arr2):
    trace_loss = go.Scatter(
        x=arr1,
        y=arr2
    )
    layout = go.Layout(
        title="Lost function followed by the number of iterations!!!!!!",
        xaxis=dict(
            title="The number of iterations"
        ),
        yaxis=dict(
            title="Lost function"
        )
    )

    fig = go.Figure(data=[trace_loss], layout=layout)
    plt.offline.plot(fig, filename='lost_function.html')
    print("The lost function is visulized in lost_function.html")


def draw_cluster(X, y, title):
    X00 = []
    X01 = []
    X10 = []
    X11 = []
    X20 = []
    X21 = []
    X30 = []
    X31 = []
    X40 = []
    X41 = []

    for i in range(len(y)):
        if int(y[i]) == 0:
            X00.append(X[0][i])
            X01.append(X[1][i])
        if int(y[i]) == 1:
            X10.append(X[0][i])
            X11.append(X[1][i])
        if int(y[i]) == 2:
            X20.append(X[0][i])
            X21.append(X[1][i])
        if int(y[i]) == 3:
            X30.append(X[0][i])
            X31.append(X[1][i])
        if int(y[i]) == 4:
            X40.append(X[0][i])
            X41.append(X[1][i])
    trace0 = go.Scatter(
        x=X00,
        y=X01,
        mode="markers",
        name='Cluster 1'
    )
    trace1 = go.Scatter(
        x=X10,
        y=X11,
        mode="markers",
        name='Cluster 2'
    )
    trace2 = go.Scatter(
        x=X20,
        y=X21,
        mode="markers",
        name='Cluster 3'
    )
    trace3 = go.Scatter(
        x=X30,
        y=X31,
        mode="markers",
        name='Cluster 4'
    )
    trace4 = go.Scatter(
        x=X40,
        y=X41,
        mode="markers",
        name='Cluster 5'
    )

    layout = go.Layout(
        title=title
    )

    trace = [trace0, trace1, trace2, trace3, trace4]
    fig = go.Figure(data=trace, layout=layout)
    plt.offline.plot(fig, filename='clusters.html')
    print("The results is visulized in clusters.html")


# Change the parameter here
X, y = read_data('bai1_input.txt')
num_labels = 5
num_hidder_layer_nodes = 100
num_iters = 5000

W1, b1, W2, b2, y, loss = train(X, y, num_labels, num_hidder_layer_nodes, num_iters)
y_predicted = predict(X, W1, b1, W2, b2, y)

# 3. Draw lost function followed by iteration number
x_axis = list(range(0, num_iters))
draw_loss(x_axis, loss)
# 3. Draw the results
draw_cluster(X, y, 'Dataset after being clustered!!!!!!!!!!')

# 4. Change the number of hidden layer (NOT FINISHED)
# def train_network(X,y,C,hidden_layer_size,num_layers,num_iters):
#     print ("The model is training .......!!!!")
#     loss = list()
#     d0 = 2 # the dimension number of X
#     d1 = hidden_layer_size
#     d2 = C
#
#     # Initialize parametters randomly
#     W1 = 0.001 * np.random.randn(d0, d1)
#     b1 = np.zeros((d1, 1))
#
#     num_hidden_layers =num_layers -2
#
#     # weight and bias of hidden layers from 2nd hidden layer
#     Wmid = list(range(num_hidden_layers))
#     bmid = list(range(num_hidden_layers))
#
#     if num_hidden_layers > 1:
#         for i in range(num_hidden_layers-1):
#             Wmid[i] = 0.001 * np.random.randn(d1, d1)
#             bmid[i] = np.zeros((d1, 1))
#
#     W2 = 0.001 * np.random.randn(d1, d2)
#     b2 = np.zeros((d2, 1))
#
#     Y = one_hot_coding(y, C)
#     N = X.shape[1] #The number of samples in train dataset
#     eta = 1 # learning rate
#
#     Zmid = list(range(num_hidden_layers))
#     Amid = list(range(num_hidden_layers))
#     Emid = list(range(num_hidden_layers))
#     dWmid = list(range(num_hidden_layers))
#     dbmid = list(range(num_hidden_layers))
#
#     for i in xrange(num_iters):
#     # 1. Feedforward to compute loss and the loss is computed here (followed the fomula)
#         Z1 = np.dot(W1.T, X) + b1
#         A1 = np.maximum(Z1, 0)
#
#         if num_hidden_layers > 1:
#             for i in range(num_hidden_layers):
#                 if i == 0:
#                     Zmid[i] = Z1
#                     Amid[i] = A1
#                 else:
#                     j = i-1
#                     tmp = np.dot(Wmid[j].T, Amid[j]) + bmid[i]
#                     tmp2 = np.maximum(tmp,0)
#
#                     Zmid[i] = tmp
#                     Amid[i] = tmp2
#             Z2 = np.dot(W2.T, Amid[num_hidden_layers-1]) + b2
#             A_last_hidden_layer = Amid[num_hidden_layers-1]
#         else:
#             Z2 = np.dot(W2.T, A1) + b2
#             A_last_hidden_layer = A1
#
#
#         Ymu = softmax(Z2)
#         loss.append(cost(Y,Ymu))
#
#     # 2. Backpropagation Step
#         E2 = (Ymu - Y)/N
#         dW2 = np.dot(A_last_hidden_layer, E2.T)
#         db2 = np.sum(E2, axis = 1, keepdims= True)
#
#         if num_hidden_layers > 1:
#             for i in range(num_hidden_layers-2, -1, -1):
#                 if i >= num_hidden_layers-2:
#                     i = num_hidden_layers-2
#                     Emid[i] = np.dot(W2,E2)
#                     Emid[i][Zmid[i]<=0] = 0
#                     dWmid[i] = np.dot(Amid[num_hidden_layers-2],Emid[i].T)
#                     dbmid[i] = np.sum(Emid[i], axis=1, keepdims=True)
#
#             #     else:
#             #         j = i+1
#             #         Emid[i] = np.dot(Wmid[j],Emid[j])
#             #         Emid[i][Zmid[i] <= 0] = 0
#             #         dWmid[i] = np.dot(Amid[i-1], Emid[i].T)
#             #         dbmid[i] = np.sum(Emid[i], axis=1, keepdims=True)
#             # #
#             E1 = np.dot(Wmid[0],Emid[0])
#         else:
#             E1 = np.dot(W2, E2)
#
#         E1[Z1<=0] = 0 # use gradient of RelU
#         dW1 = np.dot(X, E1.T)
#         db1 = np.sum(E1, axis=1, keepdims=True)
#
#     # Update Gradient Descent
#         W1 = W1 - eta*dW1
#         b1 = b1 - eta*db1
#
#         if num_hidden_layers > 1:
#             for i in range(num_hidden_layers-1):
#                 Wmid[i] = Wmid[i] - eta*dWmid[i]
#                 bmid[i] = bmid[i] - eta*dbmid[i]
#         W2 = W2 - eta*dW2
#         b2 = b2 - eta*db2
#
#     loss = np.asarray(loss)
#     print ("Finish training step.")
#     return W1,b1,W2,b2,y,loss
#
# W1,b1,W2,b2,y,loss = train_network(X, y , num_labels, num_hidder_layer_nodes,4, num_iters)
# print(loss)
