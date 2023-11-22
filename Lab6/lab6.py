import numpy as np
import matplotlib.pyplot as plt

# Ex. 1.
N = 100 # array size
values = np.array([np.random.rand() for i in range(N)])

# first x <- x * x
values1 = np.convolve(values, values)

# second x <- x * x
values2 = np.convolve(values1, values1)

# third x <- x * x
values3 = np.convolve(values2, values2)

space = np.linspace(0, 100, 100)

#  plot
fig, axs = plt.subplots(4)

for a in axs.flat:
    a.set_xlabel('index')
    a.set_ylabel('value')


axs[0].plot(values)
axs[1].plot(values1)
axs[2].plot(values2)
axs[3].plot(values3)
plt.show()

plt.savefig('Ex1.png', format='png')
plt.savefig('Ex1.pdf', format='pdf')

#%%
# Ex. 2
N = 100 # array size
space = np.linspace(0, 200, 200)
p = np.array([np.random.rand() for i in range(2 * N)]) 
q = np.array([np.random.rand() for i in range(2* N)])

# padding
for i in range(100, 200):
    p[i] = q[i] = 0

r = np.zeros(2 * N) # result
r_fft = np.zeros(2 * N) # result

# inmultirea normala a polinoamelor
for i in range(N):
   for j in range(N):
        r[i + j] += p[i] * q[j]

# inmultirea cu fft
p_fft = np.fft.fft(p)
q_fft = np.fft.fft(q)
r_fft = p_fft * q_fft

r_fft = np.fft.ifft(r_fft)

# plot
plt.plot(r)
plt.plot(r_fft)
plt.show()

plt.savefig('Ex2.png', format='png')
plt.savefig('Ex2.pdf', format='pdf')
#%%
# Ex. 3
f = 100
phi = 0
A = 1
N = 200

# sinusoidal function
def sinusodialSignal(t):
    return A * np.sin(2 * np.pi * f * t + phi)

# rectangle window
def rectangleWindow(size):
    return np.zeros(size) + 1

# Hanning window
def HanningWindow(size):
    window = np.zeros(size)
    for i in range(size):
        window[i] = 1 - np.cos(2 * np.pi * i / size)
    return window / 2


time = np.linspace(0, 1, N) # time
signal = sinusodialSignal(time) # signal

rectangle = rectangleWindow(N) # rectangle window
hanning = HanningWindow(N) # Hanning Window

rectangleSignal = signal * rectangle
hanningSignal = signal * hanning

# plot
fig, axs = plt.subplots(3)

for a in axs.flat:
    a.set_xlabel('time')
    a.set_ylabel('signal')


axs[0].plot(time, signal)
axs[1].plot(time, rectangleSignal)
axs[2].plot(time, hanningSignal)
plt.show()

plt.savefig('Ex3.png', format='png')
plt.savefig('Ex3.pdf', format='pdf')

#%% 
# Ex. 4


