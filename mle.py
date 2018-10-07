import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
mu = 25                                   # mean of distribution
sigma = 5                                 # standard deviation of distribution
x = mu + sigma * np.random.randn(10000)

def mle(x):
    u = np.mean(x)
    return u, np.sqrt(np.dot(x - u, (x - u).T) / x.shape[0])

print(mle(x))
num_bins = 100
plt.hist(x, num_bins)
plt.show()