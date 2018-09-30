import numpy as np
from sklearn.linear_model import LogisticRegression

# load data
data = np.loadtxt("data.csv", delimiter=',', skiprows=1)
X = data[:, 1:3]
y = data[:, 0].astype(int)

logR = LogisticRegression()
