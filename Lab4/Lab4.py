#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:11:51 2023

@author: anne
"""
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import time
import math

# Ex. 1

A = 1
phi = 0 # faza
f = 2 
num = 1000
start = 0 # n = 0,...,1
end = 1 # n = 0,...,1
Y = []
F = []

N = [128, 256, 512, 1024, 2048, 4096, 8192]
execTime = [[], []]
timp = []


def sinusoidalWave(t):
    return A * np.sin(2 * np.pi * f * t + phi)

def getSignal(n):
    signal = []
    # Generate time axis
    timp = np.linspace(start, end, n)
    
    # Generate signal
    for t in timp:
        signal.append(sinusoidalWave(t))
        
    return signal
        

def FourierMatrix(n):
    f = np.zeros((n, n), dtype = complex)
    for m in range(n):
        for k in range(n):
            f[m, k] = np.exp(-2 * np.pi * 1j * k * m / n)
    
    return f


# calculate time of fft vs my own func
for i in range(len(N)):
    
    # generate a signal
    signal = getSignal(N[i])
    print(len(signal))
    
    timeStart = time.time()
    Y = fft(np.array(signal))
    timeStop = time.time()
    
    # Exec Time of numpy.fft
    execTime[0].append(math.log(timeStop - timeStart))
    
    timeStart = time.time()
    F = FourierMatrix(N[i])
    Y = np.matmul(np.array(signal), F)
    timeStop = time.time()
    
    # Exec time of my func
    execTime[1].append(math.log(timeStop - timeStart))

# plot the logaritmic time
plt.xlabel("Dimeniusea vectorului")
plt.ylabel("Timpul de executie")
plt.plot(N, execTime[0])
plt.plot(N, execTime[1])
plt.show()
plt.savefig('ex1.pdf', format='pdf')
plt.savefig('ex1.png', format='png')

#%%
# Ex. 2
A = 1
phi = 0 # faza

num = 1000
start = 0 # n = 0,...,1
end = 1 # n = 0,...,1
Y = []
F = []

timp = []


def sinusoidalWave(t, fs):
    return A * np.sin(2 * np.pi * fs * t + phi)

def getSignal(fs, timp):
    signal = []
    
    # Generate signal
    for t in timp:
        signal.append(sinusoidalWave(t, fs))
        
    return signal

# Generate time axis
timp = np.linspace(start, end, num)

signal1 = getSignal(10, timp)
signal2 = getSignal(20, timp)
signal3 = getSignal(30, timp)

# Generate time axis
timp4 = np.linspace(start, end, num = 3)
signal4 = getSignal(10, timp4)

# plot Signals
fig, axs = plt.subplots(3)

    
axs[0].plot(timp, signal1)
axs[0].stem(timp4, signal4)
axs[1].plot(timp, signal2)
axs[1].stem(timp4, signal4)
axs[2].plot(timp, signal3)
axs[2].stem(timp4, signal4)

plt.show()
plt.savefig('ex2.pdf', format='pdf')
plt.savefig('ex2.png', format='png')

#%%
# Ex. 3
import numpy as np
import matplotlib.pyplot as plt
A = 1
phi = 0 # faza

num = 1000
start = 0 # n = 0,...,1
end = 1 # n = 0,...,1
Y = []
F = []

timp = []


def sinusoidalWave(t, fs):
    return A * np.sin(2 * np.pi * fs * t + phi)

def getSignal(fs, timp):
    signal = []
    
    # Generate signal
    for t in timp:
        signal.append(sinusoidalWave(t, fs))
        
    return signal

# Generate time axis
timp = np.linspace(start, end, num)

signal1 = getSignal(10, timp)
signal2 = getSignal(20, timp)
signal3 = getSignal(30, timp)

# Generate time axis
timp4 = np.linspace(start, end, num = 10)
signal4 = getSignal(10, timp4)

# plot Signals
fig, axs = plt.subplots(3)

    
axs[0].plot(timp, signal1)
axs[0].stem(timp4, signal4)
axs[1].plot(timp, signal2)
axs[1].stem(timp4, signal4)
axs[2].plot(timp, signal3)
axs[2].stem(timp4, signal4)

plt.show()
plt.savefig('ex3.pdf', format='pdf')
plt.savefig('ex3.png', format='png')

#%%
# Ex. 4
# fs > 2B => fs > 2 * 200
B = 200 
fs = 401


#%%
# Ex. 5
# Rulat in Audacity
#%% 
# Ex. 6
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

rate, x = scipy.io.wavfile.read("aeiou.wav")
N = len(x)
groups = []

# Create groups with overlap
for i in range(0, N, N // 200):
    # Create a group
    group = []
    for j in range(i, min(N, i + N // 100)):
        group.append(x[j])
    
    # Add the new group to the group list
    groups.append(group)

# Calculate FFT for each group
fft_groups = []
for group in groups:
    fft_groups.append(np.fft.fft(np.array(group)))

# fft_groups = fft_groups[: N // 2]

spectogram = np.zeros((len(groups), len(groups[0])))

# Create the spectogram
for i in range(len(fft_groups)):
    for j in range(len(fft_groups[i])):
        spectogram[i][j] = abs(fft_groups[i][j])

# Display the spectogram
plt.imshow(spectogram.T, norm='log')
plt.savefig('ex6.pdf', format='pdf')
plt.savefig('ex6.png', format='png')

#%%
# Ex. 7
# SNR = Psemnal / Pzgomot
# Psemnal = 90 dB
# SNRdB = 80 dB
#  SNRdB = 10 log10 SNR
# Pzgomot = ?

Psemnal = 90 
SNRdB = 80
SNR = 10 ** (SNRdB / 10)
Pzgomot = Psemnal / SNR
print("Puterea zgomotului este:" , math.log(Pzgomot))