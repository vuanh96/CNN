from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
import keras
from keras.callbacks import TensorBoard

# load data
data = np.loadtxt("data.csv", delimiter=',', skiprows=1)
X = data[:, 1:3]
y = data[:, 0].astype(int)
# one hot coding
y = np_utils.to_categorical(y)

# create model
model = Sequential()
model.add(Dense(100, input_shape=(2,), activation=tf.nn.relu))
model.add(Dense(3, activation=tf.nn.softmax))

# complie model
tenserboard = TensorBoard(log_dir='./logs', write_graph=1)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.adam(lr=.01),
              metrics=['accuracy'])

# fit model
model.fit(X, y, epochs=1000,
          batch_size=300,
          verbose=1,
          callbacks=[tenserboard])
# evaluate the model, acc = 99.33%
scores = model.evaluate(X, y, batch_size=300)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# tensorboard --logdir=logs --host=localhost

