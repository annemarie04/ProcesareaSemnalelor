#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:55:44 2023

@author: anne
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import scipy.signal


# Lab. 2. / Ex. 1.
A = 1
phi = 0
f = 1
sinSignal = []
cosSignal = []
start = 0
end = 1
phi2 = 3 * np.pi / 2

def sinWave(t):
    return A * np.sin(2 * np.pi * f * t + phi)

def cosWave(t):
    return A * np.cos(2 * np.pi * f * t + phi2)

time = np.linspace(start, end, num = 1000)

for i in range(0, len(time)):
    sinSignal.append(sinWave(time[i]))
    cosSignal.append(cosWave(time[i]))
    
    
fig, axs = plt.subplots(2)
fig.suptitle("Lab. 2 / Ex. 1")

plt.xlabel("Timp")
plt.ylabel("f(Timp)")

# Plot
axs[0].plot(time,sinSignal)
axs[1].plot(time, cosSignal)


    



#%%
# Lab. 2. / Ex. 2.
import math 
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile 

A = 1
phi = 0 # faza
f = 1

start = 0
end = 1

time = []
signal = [[], [], [], []]
noiseSignal = [[], [], [], []]
n = []
SNR = [0.1, 1, 10, 100]
gamma = []
normX = 0
normZ = 0
#generati z cat x; z = zgomot gaussian -> functie de zgomot gaussian in numpy
# x +c z * gamma
# cu cat gamma este mai mare cu atat este mai mult zgomot
# se afla gamma
# afisam x  + gamma * z


def sinusoidalWave(t):
    return A * np.sin(2 * np.pi * f * t + phi)

# Generate time axis
time = np.linspace(start, end, num = 1000)

# Generate gaussian noise vector
noise = np.random.normal(0, 1, 1000)

# Try 4 different sinusoidal waves
for j in range(4):
    for i in range(0, len(time)):
        phi = j + 1
        signal[j].append(sinusoidalWave(time[i]))
        
# Plot the 4 sinusoidas waves
plt.plot(time, signal[0])
plt.plot(time, signal[1])
plt.plot(time, signal[2])
plt.plot(time, signal[3])
plt.show()
        
# Calculate Norm of X
for i in signal[0]:
    normX = normX + i * i
    
# Calculate Nrom of Z
for i in noise:
    normZ = normZ + i * i
    
# Calculate gamma variables for all SNR
for i in range(len(SNR)):
    gamma.append(math.sqrt(normX / (normZ * SNR[i])))
    
    
print("Gamma:", gamma)

for j in range(4):
    for i in range(len(time)):
        noiseSignal[j].append(signal[0][i] + gamma[j] * noise[i])
        

# Plot Noise Waves
fig, axs = plt.subplots(4)
fig.suptitle("Lab. 2 / Ex. 2")

plt.xlabel("Timp")
plt.ylabel("f(Timp)")

axs[0].plot(time, noiseSignal[0])
axs[1].plot(time, noiseSignal[1])
axs[2].plot(time, noiseSignal[2])
axs[3].plot(time, noiseSignal[3])
plt.show()

# Lab. 2. / Ex. 3.
# sounddevice :)

#rate = int(10e5)
fs = 2000 # frecventa de esantionare
rate = int(10e5)

sd.play(signal[0], fs)
sd.wait()
sd.play(signal[0], fs)
sd.wait()
sd.play(signal[0], fs)
sd.wait()
sd.play(signal[0], fs)
sd.wait()

scipy.io.wavfile.write('nume.wav', rate, signal) # salvarea semnalului in format audio

rate, x = scipy.io.wavfile.read('nume.wav') # incarcarea unui semnal salvat anterior
sd.play(x, fs) # redarea semnalului audio

#%%
# Lab. 2. / Ex. 4.
f = 240 
start = 0
end = 0.05

time = []
# Sinusoidal
# f = 800
# A = 1
# phi = 0

# Sawtooth
# A = 1 # amplitude
# T = 1 / f # period of the wave
# phi = 0  # phase

def sinusoidalWave(t):
    return 1 * np.sin(2 * np.pi * 800 * t + 0)

def sawtoothWave(x):
    return 1 * ((x / ( 1 / 240) + 0) - np.floor(x / (1 / 240) + 0))

signal = [[], [], []]
time = np.linspace(start, end, num = 1000)

for i in range(0, len(time)):
    signal[0].append(sinusoidalWave(time[i]))
    signal[1].append(sawtoothWave(time[i]))
    signal[2].append(sinusoidalWave(time[i]) + sawtoothWave(time[i]))
    
fig, axs = plt.subplots(3)
fig.suptitle("Lab. 2 / Ex. 4")

plt.xlabel("Timp")
plt.ylabel("f(Timp)")

# Plot
axs[0].plot(time, signal[0])
axs[1].plot(time, signal[1])
axs[2].plot(time, signal[2])


#%%
# Lab. 2. / Ex. 5.
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
A = 1
phi = 0 # faza
fs = 1000


start = 0
end = 1

time = []
signal = []

def sinusoidalWave(t, f):
    return A * np.sin(2 * np.pi * f * t + phi)

time = np.linspace(start, end, num = 1000)

for i in range(len(time)):
    if i < len(time) / 2:
        signal.append(sinusoidalWave(time[i], 240))
    else:
        signal.append(sinusoidalWave(time[i], 480))
    

plt.xlabel("Timp")
plt.ylabel("f(Timp)")
plt.plot(time, signal)
plt.show()

sd.play(signal, 2000)
sd.wait()

#%%
# Lab. 2. / Ex. 6.
A = 1
phi = 0 # faza
fs = 1000
f = 1

start = 0
end = 1

time = []
signal = [[], [], []]

def sinusoidalWave(t):
    return A * np.sin(2 * np.pi * f * t + phi)

time = np.linspace(start, end, num = 1000)

for i in range(len(time)):
    f = fs / 2
    signal[0].append(sinusoidalWave(time[i]))
    f = fs / 4
    signal[1].append(sinusoidalWave(time[i]))
    f = 0
    signal[2].append(sinusoidalWave(time[i]))
    
plt.plot(time, signal[0])
plt.plot(time, signal[1])
plt.plot(time, signal[2])
plt.show()

fig, axs = plt.subplots(3)
fig.suptitle("Lab. 2 / Ex. 6")

plt.xlabel("Timp")
plt.ylabel("f(Timp)")

axs[0].plot(time, signal[0])
axs[1].plot(time, signal[1])
axs[2].plot(time, signal[2])
plt.show()



#%%
# Lab. 2. / Ex. 7.
A = 1 # amplitudine 
phi = 0 # faza
f = 200

start = 0
end = 1

time = [[], [], [], []]
signal = [[], [], [], []]

def sinusoidalWave(t):
    return A * np.sin(2 * np.pi * f * t + phi)

time[0] = np.linspace(start, end, num = 1000)

for i in range(len(time[0])):
    signal[0].append(sinusoidalWave(time[0][i]))
    
# Prima decimare
for i in range(len(time[0])):
    if i % 4 == 0:
        signal[1].append(sinusoidalWave(time[0][i]))
        time[1].append(time[0][i])
    if i % 4 == 1:
        signal[2].append(sinusoidalWave(time[0][i]))
        time[2].append(time[0][i])
    
fig, axs = plt.subplots(3)
fig.suptitle("Lab. 2 / Ex. 7")

plt.xlabel("Timp")
plt.ylabel("f(Timp)")

# Plot
axs[0].plot(time[0], signal[0])
axs[1].plot(time[1], signal[1])
axs[2].plot(time[2], signal[2])

plt.show()


#%%
# Lab. 2. / Ex. 8.
import numpy as np
import matplotlib.pyplot as plt

start = -np.pi / 2
end = np.pi / 2

def Pade(x):
    return (x - 7 * x ** 3 / 60) / ( 1 + x ** 2 / 20)

time = np.linspace(start, end, num = 1000)
signal = [[], [], []]
for i in range(len(time)):
    signal[0].append(np.sin(time[i]))
    signal[1].append(Pade(time[i]))
    signal[2].append(signal[0][i] - signal[1][i])

plt.xlabel("Timp")
plt.ylabel("Sin(t)")

plt.plot(time, signal[0])
plt.plot(time, signal[1])
plt.show()

plt.plot(time, signal[2])
plt.show()
# %%
