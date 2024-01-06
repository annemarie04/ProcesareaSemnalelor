from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

N = 1000
time = np.linspace(1, 1000, num = 1000)

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
eroare = np.random.rand(1000)
qOrizont = 5
m = 100

mean = np.sum(serieDeTimp) / 700
sample = np.zeros(m)
Y = np.zeros((m, qOrizont))

for i in range(m):
    sample[i] = serieDeTimp[700 - m + i] - serieDeTimp[qOrizont + i] - mean



for i in range(m):
    Y[i] = eroare[i : i + qOrizont]

teta = np.linalg.lstsq(Y, sample, rcond=None)[0]
print(teta)

# Ex. 4
ARMA = ARIMA(serieDeTimp[:700],
             order=([p for p in range(21)], 0, [q for q in range(21)]),
             trend = 'ct')
ARMA = ARMA.fit()
predictions = ARMA.forecast(300)

eroare = np.mean((predictions - serieDeTimp[700: 1000]) ** 2)
print(eroare)