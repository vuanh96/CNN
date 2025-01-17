from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

N = 100  # number of points per class
d0 = 2  # dimensionality
C = 3  # number of classes
X = np.zeros((d0, N * C))  # data matrix (each row = single example)
y = np.zeros(N * C, dtype='uint8')  # class labels

fp = open("data.csv", "w")
fp.write("label, x1, x2\n")
for j in range(C):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[:, ix] = np.c_[r * np.sin(t), r * np.cos(t)].T
    y[ix] = j
    for i in ix:
        fp.write("%d,%.16f,%.16f\n" % (y[i], X[0][i], X[1][i]))

fp.close()


# lets visualize the data:
plt.plot(X[0, :N], X[1, :N], 'bs', markersize=7)
plt.plot(X[0, N:2 * N], X[1, N:2 * N], 'ro', markersize=7)
plt.plot(X[0, 2 * N:], X[1, 2 * N:], 'g^', markersize=7)
# plt.axis('off')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.show()
