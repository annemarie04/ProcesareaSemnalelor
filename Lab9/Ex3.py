from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa as sm

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

# model_ma = sm.arma_process(serieDeTimp, order=(0, qOrizont))
# result = model_ma.fit()
sm.ArmaProcess.arma2ma(lags=None)
# print(result.summary())