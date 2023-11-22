import numpy as np
import matplotlib.pyplot as plt

# Ex. 4 a)
train = np.genfromtxt('Train.csv', delimiter=',')
values = train[:,2] # get the data
x = values[1:] # exclude first ID, Datetime, Count
x = x[:73] # get values for 3 days

# Ex. 4 b)
w_values = [5, 9, 13, 17]

# plot
fig, axs = plt.subplots(len(w_values))

for a in axs.flat:
    a.set_xlabel('time')
    a.set_ylabel('signal')

for i in range(len(w_values)):
    w = w_values[i]
    x_w = np.convolve(x, np.ones(w), 'valid') / w
    axs[i].plot(np.linspace(0, len(x_w) - 1, len(x_w)), x_w)

plt.show()
plt.savefig('Ex4_b.png', format='png')
plt.savefig('Ex4_b.pdf', format='pdf')

# Ex. 4 c)
# Frecventa esantionare = 1 / 3600
# Frecventa maxima = frecventa esantionare / 2 = 1 / 7200
# Frecventa aleasa = 1 / 9000 
# 0  -------    x     ------ 1
# 0  ------- 1 / 9000 ------ 1 / 7200
# x = (1 / 9000) / (1 / 7200) = 7200 / 9000 = 4/5 = 0.8 (Frecventa normalizata)

# Ex. 4 d)
import scipy
ordFilt = 5
rp = 5
Wn = 0.8 # de la c)
butterB, butterA = scipy.signal.butter(ordFilt, Wn)
chebyB, chebyA = scipy.signal.cheby1(ordFilt, rp, Wn)

plt.xlabel('Frecventa')
plt.ylabel('Valoare')

w, h = scipy.signal.freqz(butterB, butterA) 
plt.plot(w, 20 * np.log10(abs(h))) # afisare cu scara logaritmica

w, h = scipy.signal.freqz(chebyB, chebyA)
plt.plot(w, 20 * np.log10(abs(h))) # afisare cu scara logaritmica

plt.show()

plt.savefig('Ex4_d.png', format='png')
plt.savefig('Ex4_d.pdf', format='pdf')

# Ex. 4 e)
time = np.linspace(0, len(x) - 1, len(x))

x_Butter = scipy.signal.filtfilt(butterB, butterA, x)
x_Cheby = scipy.signal.filtfilt(chebyB, chebyA, x)

# plot
fig, axs = plt.subplots(2)
for a in axs.flat:
    a.set_xlabel('time')
    a.set_ylabel('signal')

axs[0].plot(time, x)
axs[0].plot(time, x_Butter)
axs[1].plot(time, x)
axs[1].plot(time, x_Cheby)
plt.show()

plt.savefig('Ex4_e.png', format='png')
plt.savefig('Ex4_e.pdf', format='pdf')
#Aleg Butterworth deoarece este mai apropiat de semnalul dat

# Ex. 4 f)
# Create Filtera
butterFilters = [
                scipy.signal.butter(2, Wn),
                scipy.signal.butter(5, Wn),
                scipy.signal.butter(8, Wn)
                ]
chebyFilters = [scipy.signal.cheby1(2, 5, Wn),
                 scipy.signal.cheby1(5, 1, Wn),
                 scipy.signal.cheby1(8, 4, Wn),
                 scipy.signal.cheby1(2, 2, Wn),
                 scipy.signal.cheby1(5, 10, Wn),
                 scipy.signal.cheby1(8, 8, Wn)
                 ]

# Plot Butter
fig, axs = plt.subplots(3)
for a in axs.flat:
    a.set_xlabel('Frecventa')
    a.set_ylabel('Valoare')

i_axs = 0

for (butterB, butterA) in butterFilters:
    w, h = scipy.signal.freqz(butterB, butterA)
    axs[i_axs].plot(w, 20 * np.log10(abs(h)))
    i_axs += 1

plt.show()   
plt.savefig('Ex4_f_Butter.png', format='png')
plt.savefig('Ex4_f_Butter.pdf', format='pdf')

# Plot Cheby  
fig, axs = plt.subplots(6)
for a in axs.flat:
    a.set_xlabel('Frecventa')
    a.set_ylabel('Valoare')

i_axs = 0

for (chebyB, chebyA) in chebyFilters:
    w, h = scipy.signal.freqz(chebyB, chebyA)
    axs[i_axs].plot(w, 20 * np.log10(abs(h)))
    i_axs += 1
plt.show()    
plt.savefig('Ex4_f_Cheby.png', format='png')
plt.savefig('Ex4_f_Cheby.pdf', format='pdf')
