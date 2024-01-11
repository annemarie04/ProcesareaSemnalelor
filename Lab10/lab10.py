import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal

N = 1000
# Unidimensional Gaussian Distribution
gaussianUnidim = np.random.normal(loc = 1, scale = 2, size = ( N))

# plt.hist(gaussianUnidim)
# plt.show()



# Bidimensional Gaussian Distribution
# cov_vals = [1, 3/5, 2]
cov = np.array([[1, 3/5], [3/5, 2]])
mean = np.array([0, 0])
plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize']=10,6

    
distr = multivariate_normal(cov = cov, mean = mean,
                                seed = N)

# Sampling
data = distr.rvs(size = 5000)

# Plotting the generated samples
plt.plot(data[:,0],data[:,1], 'o', c='lime',
             markeredgewidth = 0.5,
             markeredgecolor = 'black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')

plt.show()