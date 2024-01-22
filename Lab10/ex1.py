import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal

N = 1000
# Unidimensional Gaussian Distribution
gaussianUnidim = np.random.normal(loc = 1, scale = 2, size = ( N))

plt.hist(gaussianUnidim)
plt.show()
plt.savefig('unidim_gaussian_distrib.pdf', format='pdf')
plt.savefig('unidim_gaussian_distrib.png', format='png')


# Bidimensional Gaussian Distribution
# cov_vals = [1, 3/5, 2]
cov = np.array([[1, 3/5], [3/5, 2]])
mean = np.matrix([0, 0]).T
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=10,6

    
# distr = multivariate_normal(cov = cov, mean = mean,
#                                 seed = N)

# # Sampling
# data = distr.rvs(size = 5000)
d = mean.shape[0]
eig = np.linalg.eig(cov)
diag = np.zeros((d, d))

for i in range(d):
    diag[i, i] = np.sqrt(eig[0][i])

def sample():
    u = eig[1].T
    n = np.matrix(np.random.randn(d)).T
    x = u @ diag @ n + mean
    return x

data = np.array([sample() for _ in range(N)])

# Plot
plt.plot(data[:, 0, :], data[:, 1, :], 'o', c='lime',
             markeredgewidth = 0.5,
             markeredgecolor = 'black')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('auto')

plt.show()
plt.savefig('bidim_gaussian_distrib.pdf', format='pdf')
plt.savefig('bidim_gaussian_distrib.png', format='png')