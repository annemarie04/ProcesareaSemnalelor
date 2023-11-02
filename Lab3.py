#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:08:35 2023

@author: anne
"""
import numpy as np
import matplotlib.pyplot as plt
import math

N = 8
F = np.zeros((N, N), dtype = complex)

#Ex. 1

# generate the Fourier Matrix
def FourierMatrix():
    for m in range(N):
        for k in range(N):
            F[m, k] = np.exp(-2 * np.pi * 1j * k * m / N)
 
def plotMatrix():
  # get the number of rows and columns of the matrix
  rows, cols = F.shape
  # create a figure with the desired size
  plt.figure(figsize = (10, 10))
  
  # oop through each row of the matrix
  for i in range(rows):
    real = []
    imag = []
    # create a subplot for the current row
    plt.subplot(rows, 1, i + 1)
    
    # extract the real and imaginary part of the current row
    real = F[i].real
    imag = F[i].imag
    
    # display the real and imaginary part for a line
    plt.plot(real, color='blue', label='Real')
    plt.plot(imag, color='red', label='Imaginary')
    
    # add a title and a legend for the subplot
    plt.title(f'Line {i + 1}')
    plt.legend()
    
  # display the figure
  plt.show()
  
  # save figure
  plt.savefig('Ex1.pdf', format='pdf')
  plt.savefig('Ex1.png', format='png')

FourierMatrix()
print(F)
plotMatrix()

# check if the matrix is unitary
isUnitary = np.allclose(np.eye(N), np.dot(F, F.conj().T))
if isUnitary:
    print("Matricea Fourier este unitara")
else:
    print("Matricea Fourier nu este unitara")



#%%
# Ex. 2
N = 8
omega = [1, 2, 5, 7]

signal = []

A = 1
phi = 0 # faza
f = 2 
num = 1000
start = 0 # n = 0,...,1
end = 1 # n = 0,...,1
Y = []

# Generate time axis
time = np.linspace(start, end, num)

def sinusoidalWave(t):
    return A * np.sin(2 * np.pi * f * t + phi)

def getSignal():
    # Generate signal
    for t in time:
        signal.append(sinusoidalWave(t))

# Calc Y pt Cerul Unitate
def getCercUnit():
    y = np.array([signal[j] * np.exp(-2 * np.pi * 1j * j / num) for j in range(num)], dtype = complex)
    return y

# Calc Y pt mai multe omega
def getImagReal(om):
    y = np.array([signal[j] * np.exp(-2 * np.pi * 1j * om * j / num) for j in range(num)], dtype = complex)
    return y

# Plot Figure 1.
def plotFig1():
    fig, axs = plt.subplots(2)
    fig.suptitle('Ex 2 / Fig 1') 
    
    for ax in axs.flat:
        ax.set_xlabel('real')
        ax.set_ylabel('imaginar')
        
    axs[0].plot(time, signal)
    axs[1].scatter(Yunit.real, Yunit.imag, c = abs(Yunit))
    plt.show()
    
    # save figure
    plt.savefig('Ex2_Fig1.pdf', format='pdf')
    plt.savefig('Ex2_Fig1.png', format='png')

# Plot Figure 2
def plotFig2():
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Ex 2 / Fig 2') 
    
    for ax in axs.flat:
        ax.set_xlabel('real')
        ax.set_ylabel('imaginar')
        
    axs[0,0].scatter(Y[0].real, Y[0].imag, c = abs(Y[0]))
    axs[0,1].scatter(Y[1].real, Y[1].imag, c = abs(Y[1]))
    axs[1,0].scatter(Y[2].real, Y[2].imag, c = abs(Y[2]))
    axs[1,1].scatter(Y[3].real, Y[3].imag, c = abs(Y[3]))
    
    # save figure
    plt.savefig('Ex2_Fig2.pdf', format='pdf')
    plt.savefig('Ex2_Fig2.png', format='png')
    plt.show()

getSignal()

for i in range(len(omega)):
    Y[i] = getImagReal(omega[i])
    
Yunit = getCercUnit()
 
plotFig1()  
plotFig2()  
        
#%%
# Ex 3

N = 100
omega = [1, 2, 5, 7]

signal = []

A = 1
phi = 0 # faza
frecv = [4, 20, 50]
num = 1000
start = 0 # n = 0,...,1
end = 1 # n = 0,...,1
Y = []
omega = [i for i in range(N)]

# Generate time axis
time = np.linspace(start, end, num)
F = np.zeros((N, N), dtype = complex)

#Ex. 1

# generate the Fourier Matrix
def FourierMatrix():
    for m in range(N):
        for k in range(N):
            F[m, k] = np.exp(-2 * np.pi * 1j * k * m / N)
            
def sinusoidalWave(t, f):
    return A * np.sin(2 * np.pi * f * t + phi)

def getSignal():
    # Generate signal
    for t in time:
        signal.append(sinusoidalWave(t, frecv[2]) - sinusoidalWave(t, frecv[1]) - sinusoidalWave(t, frecv[0]))
            
def plotSignal():
    plt.xlabel("Timp(s)")
    plt.ylabel("x(t)")
    plt.plot(time, signal)
    plt.savefig('Ex3_Fig3_1.pdf', format='pdf')
    plt.savefig('Ex3_Fig3_1.png', format='png')
    plt.show()
    
def plotFrecv():
    plt.xlabel("Frecventa(Hz)")
    plt.ylabel("|X(omega)|")
    plt.stem(omega, [abs(X[i]) for i in range(N)])

    plt.savefig('Ex3_Fig3_2.pdf', format='pdf')
    plt.savefig('Ex3_Fig3_2.png', format='png')

getSignal()
plotSignal()

FourierMatrix()
signalArray = np.array(signal)
X = np.dot(F, signalArray.T[:100])
plotFrecv()

