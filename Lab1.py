#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 18:19:22 2023

@author: anne
"""

import numpy as np 
import matplotlib.pyplot as plt


# Ex. 1
# x(t) = cos(520πt + π/3), y(t) = cos(280πt − π/3), z(t) = cos(120πt + π/3).

def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t): 
    return np.cos(280 * np.pi * t - np.pi / 3) 

def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

def timeAxs(start, end, step):
    time = []
    while start <= (end + step):
        time.append(start)
        start += step

    return time
        
timp = timeAxs(0, 0.03, 0.0005)
X = []
Y = []
Z = []

for i in timp:
    X.append(x(i))
    Y.append(y(i))
    Z.append(z(i))

fig, axs = plt.subplots(3)
fig.suptitle("Lab. 1 / Ex. 1 / a & b")

plt.xlabel("Timp")
plt.ylabel("f(Timp)")

# Plot
axs[0].plot(timp,X)
axs[1].plot(timp,Y)
axs[2].plot(timp,Z)

Xdiscret = []
Ydiscret = []
Zdiscret = []
esantionTimp = []
# 0 -> 0.03 => 60
print(len(X))

i = 0
while i <= 60:
    esantionTimp.append(timp[i])
    
    Xdiscret.append(X[i])
    Ydiscret.append(Y[i])
    Zdiscret.append(Z[i])
    i += 12
    
print(esantionTimp)
figC, axsC = plt.subplots(3)
figC.suptitle("Lab. 1 / Ex. 1 / c")

plt.xlabel("Timp")
plt.ylabel("f(Timp)")

# Stem & plot
axs[0].stem(esantionTimp,Xdiscret)
axs[1].stem(esantionTimp,Ydiscret)
axs[2].stem(esantionTimp,Zdiscret)

# Stem            
axsC[0].stem(esantionTimp,Xdiscret)
axsC[1].stem(esantionTimp,Ydiscret)
axsC[2].stem(esantionTimp,Zdiscret)
    
#%%
# Lab. 1 / Ex. 2 a) 
# f = 400 Hz ; 1600 esantioane 
timp = []
semnal = []
start = 0
end = 1
step = 0.000625

f = 400
A = 1
phi = 0

def timeAxs(start, end, step):
    time = []
    while start <= (end + step):
        time.append(start)
        start += step
    return time

def semnalSinusoidal(t):
    return A * np.sin(2 * np.pi * f * t + phi)
    
timp = timeAxs(start, end, step)

for i in range(0, len(timp)):
    semnal.append(semnalSinusoidal(timp[i]))

plt.xlabel("Timp")
plt.ylabel("Semnal")
plt.stem(timp[:800], semnal[:800])

#%%
# Lab. 1 . Ex. 2 b)
# semnal sinusoidal; f = 800 Hz; 3 secunde

timp = []
semnal = []
start = 0
end = 3
step = 0.0001

f = 800
A = 1
phi = 0

def timeAxs(start, end, step):
    time = []
    while start <= (end + step):
        time.append(start)
        start += step
    return time

def semnalSinusoidal(t):
    return A * np.sin(2 * np.pi * f * t + phi)
        
timp = timeAxs(start, end, step)

for i in range(0, len(timp)):
    semnal.append(semnalSinusoidal(timp[i]))

plt.xlabel("Timp")
plt.ylabel("Semnal")
plt.stem(timp[:100], semnal[:100])

#%% 
# Lab. 1. Ex. 2. c)
# semnal sawtooth; f = 240 Hz
f = 240 
start = 0
end = 0.05

timp = []
semnal =[]

A = 1 # amplitude
T = 1 / f # period of the wave
phi = 0  # phase

def sawtoothWave(x):
    return A * ((x / T + phi) - np.floor(x / T + phi))

timp = np.linspace(start, end, num = 1000)

for i in range(0, len(timp)):
    semnal.append(sawtoothWave(timp[i]))

plt.xlabel("Timp")
plt.ylabel("Semnal")
plt.plot(timp, semnal)

#%%
# Lab. 1. Ex. 2. d)
# semnal square; f = 300 Hz
f = 300
start = 0
end = 0.05

timp = []
semnal =[]

A = 1 # amplitude
T = 1 / f # period of the wave
phi = 0  # phase

def squareWave(x):
    return A * np.sign(np.sin(2 * np.pi * x / T))

timp = np.linspace(start, end, num = 1000)

for i in range(0, len(timp)):
    semnal.append(squareWave(timp[i]))

plt.xlabel("Timp")
plt.ylabel("Semnal")
plt.plot(timp, semnal)

#%%
# Lab. 1. Ex. 2. e)

x = 128
y = 128

semnal = np.array(np.random.rand(x, y))
plt.imshow(semnal)


#%% 
# Lab. 1. Ex. 2. f)
n = 128

def wave(x, y):
    return x * 5 + y * 9


semnal = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        semnal[i, j] = wave(i, j)

plt.imshow(semnal)

#%% 
# Lab. 1. Ex. 3. 
f = 2000 # 2000 esantioane / 1 sec
interval = 1 / (f - 1)
print("Intervalul de timp dintre 2 esantoane este:", interval)

nr_esantioane = 60 * 60 * f # o ora de achizitie = 2000 esant /sec * 60 sec * 60 min
dim_biti = nr_esantioane * 4 # un esantion este memorat pe 4 biti
dim_bytes = dim_biti / 8 # 1 byte = 8 biti
print("O ora de achizitie va ocupa:", dim_bytes, "bytes")

