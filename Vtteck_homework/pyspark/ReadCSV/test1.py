import matplotlib.pyplot as plt

x = [1,20,50]
y1 = [23,42.5,73.5]
y2 = [34.6,386.8,904.6]

plt.plot(x, y2, 'ro-', label='no persist')
plt.plot(x, y1, 'bo-', label='cache()')
plt.legend()
plt.xlabel('Number of movies')
plt.ylabel('Time (s)')
plt.show()