from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
import math
# Ex. 2




# Calculate SNR
def calcSNR(x):
    media_imaginii = np.mean(x)
    deviatia_imaginii = np.std(x)
    snr = media_imaginii / deviatia_imaginii
    return snr


X = misc.face(gray=True) # imaginea cu ratonul

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))
snr = calcSNR(X)
snrValues = []

for freq_cutoff in range(150, 100, -1):
    Y_cutoff = Y.copy()
    Y_cutoff[freq_db > freq_cutoff] = 0
    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)   
    
    snrValues.append(calcSNR(X_cutoff))

plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.show()
plt.savefig('Ex2.png', format='png')
plt.savefig('Ex2.pdf', format='pdf')
print(snrValues)