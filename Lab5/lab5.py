import numpy as np
import matplotlib.pyplot as plt

# Ex. 1. a) 
# Numarul masinilor a fost masura din ora in ora
# => frecventa de esantionare = 1 esantion / 1 h = 1 esantion / 3600 s = 1/3600 Hz = 0,000278 Hz

# Ex. 1. b)
# Start Datetime: 25-08-2012 00:00
# End Datetime: 25-09-2014 23:00
# Time Interval: 761 days, 23 hours = 2 years, 1 month, 23 hours

# Ex. 1. c)
# Frecventa maxima = Frecventa esantionare / 2 = 1 / 7200 Hz

# Ex. 1. d)
train = np.genfromtxt('Train.csv', delimiter=',')
values = train[:,2] # get the data
values = values[1:] # exclude first "an

# print(values)
N = values.shape[0] # get the number of elements

# calculate fft
fftValues = np.fft.fft(values)

# calculate absolute value of the fft
fftAbsValues = abs(fftValues / N)

# use only half of the values
fftAbsValues = fftAbsValues[: (N // 2)]

# generate the frequencies vector
fq = 1 / 3600
frequencies = fq * np.linspace(0, N / 2, N // 2) / N

# plot 
plt.xlabel('frecventa')
plt.ylabel('fft')
plt.stem(frequencies, fftAbsValues)
plt.show()

plt.savefig('Lab5_1_d.png', format='png')
plt.savefig('Lab5_1_d.pdf', format='pdf')

# Ex. 1. e)
# calculate the average value
avg = np.mean(values)
# x[0] != 0, so there is a DC offset

# removing the offset
values = values - avg

# calculate fft
fftValues = np.fft.fft(values)

# calculate absolute value of the fft
fftAbsValues = abs(fftValues / N)

# use only half of the values
fftAbsValues = fftAbsValues[:N // 2]
avg = np.average(values)

# generate the frequencies vector
fq = 1 / 3600
frequencies = fq * np.linspace(0, N / 2, N // 2) / N

# plot 
plt.xlabel('frecventa')
plt.ylabel('fft')
plt.stem(frequencies, fftAbsValues)
plt.show()

plt.savefig('Lab5_1_e.png', format='png')
plt.savefig('Lab5_1_e.pdf', format='pdf')

# Ex. 1. f)
frIndexes = fftValues.argsort()[-4:][::-1] # get the index of the 5 values
print(frIndexes)

# get the frequencies
#print(frequencies[frIndexes])

# Ex. 1. g)
# 26-05-2014 Monday -> 15340

# get the timespan and the values 
monthValues = []
esantionStart = 15340
esantionStop = esantionStart + 24 * 30

for i in range(esantionStart, esantionStop, 1):
    monthValues.append(values[i])
print(len(monthValues))
monthValues = np.array(monthValues)
time = np.linspace(0, 720, 720)

# plot
plt.plot(time, monthValues)
plt.xlabel('Timp')
plt.ylabel('Valoare')

plt.savefig('Lab5_1_g.png', format='png')
plt.savefig('Lab5_1_g.pdf', format='pdf')

# Ex. 1. h)
# Se poate analiza evolutia in timp a semnalului si astfel se poate observa o periodicitate a datelor in functie de
# zi/noapte, sezon, an
# Se poate stabili sezonul in functie de nivelul circulatiei (mai multe masini iarna, mai putine vara)
# Intervalul de timp zi/noapte poate fi de asemnea determinat (mai putine masini noaptea)
# Aceasta analiza poate fi in schimb impiedicata de evenimente ce pot perturba datele cum ar fi demonstratii stradale, accidente
# conditii meteorologice exceptionale

# Ex. 1. i)