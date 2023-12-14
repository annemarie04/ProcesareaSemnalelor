from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

N = 1000
time = np.linspace(1, 1000, num = 1000)

# a)
# 3 componente: trend, sezon, variatii mici
# trend -> ex de gr 2
# sezon -> 2 frecv
# variatii mici -> zgomot alb gausian

def Trend():
    return np.array([1 * x**2 + 3 * x - 108 for x in time])

def Sezon():
    f1 = 3
    f2 = 7
    return np.sin(2 * np.pi * f1 * time) + np.sin(2 * np.pi * f2 * time)

def VariatiiMici():
    return np.random.normal(0.0, 0.3, size = N)

trend = Trend()
sezon = Sezon()
variatiiMici = VariatiiMici()

def SerieDeTimp():
    return trend + sezon + variatiiMici

serieDeTimp = SerieDeTimp()

# plot
fig, axs = plt.subplots(4)

axs[0].set_ylabel('Trend')
axs[0].set_xlabel('Timp')
axs[0].plot(time, trend)

axs[1].set_ylabel('Sezon')
axs[1].set_xlabel('Timp')
axs[1].plot(time, sezon)

axs[2].set_ylabel('Variatii Mici')
axs[2].set_xlabel('Timp')
axs[2].plot(time, variatiiMici)

axs[3].set_ylabel('Serie de Timp')
axs[3].set_xlabel('Timp')
axs[3].plot(time, serieDeTimp)
plt.show()

plt.savefig('Ex1_a.png', format='png')
plt.savefig('Ex1_a.pdf', format='pdf')

# Ex. 2
alphaVals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
alphaSumResults = []

# Medierea exponentiala
def med_exp(serieDeTimp, alpha):
    resultS = [serieDeTimp[0]] # rezultatele medierii
    for t in range(1, len(serieDeTimp)):
        newVal = alpha * serieDeTimp[t] + (1 - alpha) * resultS[t - 1]
        resultS.append(newVal)
    return resultS

# minSum
def minSum(origSeries, resultSeries):
    sum = 0
    for t in range(0, len(serieDeTimp) - 1):
        sum += (resultSeries[t] - origSeries[t + 1]) ** 2
    return sum

# Testing alphas
for alpha in alphaVals:
    resultSeries = med_exp(serieDeTimp, alpha)
    resultSum = minSum(serieDeTimp, np.array(resultSeries))
    alphaSumResults.append([alpha, resultSum, resultSeries])

sortedAlphaSumResults = sorted(alphaSumResults, key=lambda x: x[1])

plt.plot(time, serieDeTimp)
plt.plot(time, sortedAlphaSumResults[0][2])
plt.show()