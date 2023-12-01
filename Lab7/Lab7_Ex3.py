import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage

# Calculate SNR
def calcSNR(x):
    media_imaginii = np.mean(x)
    deviatia_imaginii = np.std(x)
    snr = media_imaginii / deviatia_imaginii
    return snr


pixel_noise = 200
X = misc.face(gray=True) # imaginea cu ratonul
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')
plt.show()

Y = np.fft.fft2(X_noisy)
freq_db = 20*np.log10(abs(Y))
X_cutoff = np.fft.ifft2(Y)
X_cutoff = np.real(X_cutoff)
print(calcSNR(X_cutoff))

snrValues = []

for freq_cutoff in range(150, 100, -1):
    Y_cutoff = Y.copy()
    Y_cutoff[freq_db > freq_cutoff] = 0

    X_cutoff = np.fft.ifft2(Y_cutoff)
    X_cutoff = np.real(X_cutoff)
    snrValues.append([freq_cutoff, calcSNR(X_cutoff)])

plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.show()
print(snrValues)