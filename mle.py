import numpy as np
import matplotlib.pyplot as plt

# =======================================================================
# This short program implements the Minimum Likelihood Estimation
# mu and sigma can be assigned according to the demand of the user
# =======================================================================

fig = plt.figure()
mu = 25                         # the mean value of the distribution
sigma = 5                       # the standard deviation of the distribution
x = mu + sigma * np.random.randn(10000)

def mle(x):
    u = np.mean(x)
    return u, np.sqrt(np.dot(x - u, (x - u).T) / x.shape[0])

print(mle(x))
num_bins = 100
plt.hist(x, num_bins)
plt.show()
