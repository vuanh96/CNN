# import numpy as np
#
# # read data
# dataset = np.genfromtxt('input.txt', dtype='int', delimiter='', skip_header=1)
# print(np.unique(dataset[:,0]))
#
# from keras.utils import np_utils
# import plotly
# import plotly.graph_objs as go
#
# plotly.offline.plot({
#     "data": [go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
#     "layout": go.Layout(title="hello world")
# }, auto_open=True)

import matplotlib.pyplot as plt
import numpy as np

import plotly
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

gaussian_numbers = np.random.randn(1000)
plt.hist(gaussian_numbers)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

fig = plt.gcf()

plot_url = plotly.offline.plot_mpl(fig, filename='mpl-basic-histogram.html')
