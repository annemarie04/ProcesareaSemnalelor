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
    return np.array([4 * x**2 + 13 * x - 8 for x in time])

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

# b)
# Calculul autocorelatiei folosind numpy
autocorr = np.correlate(serieDeTimp, serieDeTimp, mode='full')
plt.plot(autocorr)
plt.show()

plt.savefig('Ex1_b.png', format='png')
plt.savefig('Ex1_b.pdf', format='pdf')

# c)
# Parametrii pentru modelul AR
lag = 10  # Dimensiunea modelului AR
train_data = serieDeTimp[:-lag]  # Datele de antrenament

# Model AR
model = AutoReg(train_data, lags = lag)
model_fit = model.fit()

# Predicții folosind modelul AR
predictions = model_fit.predict(start = lag, end = len(serieDeTimp) - 1)

# Plot
plt.figure(figsize = (8, 4))
plt.plot(serieDeTimp, label='Serie de timp')
plt.plot(predictions, label='Predictii AR', linestyle='--')
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig('Ex1_c.png', format='png')
plt.savefig('Ex1_c.pdf', format='pdf')

# d)
best_lag = 0
best_horizon = 0
best_rmse = 0

train_data = serieDeTimp[:-50]  # Datele de antrenament
test_data = serieDeTimp[-50:] # Datele de test

for lag in (1, 50):
    for horizon in (1, 50):
        model = AutoReg(train_data, lags=lag)
        model_fit = model.fit()

        predictions = model_fit.predict(start = len(train_data), end = len(train_data) + horizon - 1)

        rmse = np.sqrt(mean_squared_error(test_data[:horizon], predictions))

        # Actualizarea cei mai buni parametrii
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag = lag
            best_horizon = horizon

print("Best lag:" + str(best_lag))
print("Best horizon:" + str(best_lag))
