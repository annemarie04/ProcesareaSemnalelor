import numpy as np
import matplotlib.pyplot as plt 

def liniar(x, y):
    return x * y


def brownian(s, t):
    return min(s, t)


def exponentiala_patrata(x, y, alpha=100):
    return np.exp(-alpha * np.linalg.norm(x - y) ** 2)


def ornstein_uhlenbeck(s, t, alpha=10):
    return np.exp(-alpha * np.abs(s - t))


def periodic(x, y, alpha=1, beta=1):
    return np.exp(-alpha * np.sin(beta * np.pi * (x - y)) ** 2)


def simetric(x, y, alpha=5):
    return np.exp(-alpha * (min(np.abs(x - y), np.abs(x + y))) ** 2)


time = np.linspace(-1, 1, 1000)

# Liniar
cov = np.array([[liniar(x, y) for y in time] for x in time])
data = np.random.multivariate_normal(mean = np.zeros(time.shape[0]), cov = cov)

plt.plot(time, data)
plt.show()
plt.savefig(f"liniar.pdf", format="pdf")
plt.savefig(f"liniar.png", format="png")

# Browniana
cov = np.array([[brownian(x, y) for y in time] for x in time])
data = np.random.multivariate_normal(mean = np.zeros(time.shape[0]), cov = cov)

plt.plot(time, data)
plt.show()
plt.savefig(f"brownian.pdf", format="pdf")
plt.savefig(f"brownian.png", format="png")

# Exponentiala Patrata
cov = np.array([[exponentiala_patrata(x, y) for y in time] for x in time])
data = np.random.multivariate_normal(mean = np.zeros(time.shape[0]), cov = cov)

plt.plot(time, data)
plt.show()
plt.savefig(f"exponentiala_patrata.pdf", format="pdf")
plt.savefig(f"exponentiala_patrata.png", format="png")

# Ornstein-Uhlenbeck
cov = np.array([[ornstein_uhlenbeck(x, y) for y in time] for x in time])
data = np.random.multivariate_normal(mean = np.zeros(time.shape[0]), cov = cov)

plt.plot(time, data)
plt.show()
plt.savefig(f"ornstein_uhlenbeck.pdf", format="pdf")
plt.savefig(f"ornstein_uhlenbeck.png", format="png")

# Periodic
cov = np.array([[periodic(x, y) for y in time] for x in time])
data = np.random.multivariate_normal(mean = np.zeros(time.shape[0]), cov = cov)

plt.plot(time, data)
plt.show()
plt.savefig(f"periodic.pdf", format="pdf")
plt.savefig(f"periodic.png", format="png")

# Simetric
cov = np.array([[simetric(x, y) for y in time] for x in time])
data = np.random.multivariate_normal(mean = np.zeros(time.shape[0]), cov = cov)

plt.plot(time, data)
plt.show()
plt.savefig("simetric.pdf", format="pdf")
plt.savefig("simetric.png", format="png")