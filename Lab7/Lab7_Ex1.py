from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

# Ex. 1
def X1(n1, n2):
    return np.sin(2 * np.pi * n1 + 3 * np.pi * n2)

def X2(n1, n2):
    return np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)

def getImage1():
    for i in range(N):
        for j in range(N):
            Y1[i][j] = X1(i, j)

def getImage2():
    for i in range(N):
        for j in range(N):
            Y2[i][j] = X2(i, j)

def getImage3():
    for i in range(N):
        for j in range(N):
            Y3[i][j] = 0
    Y3[0][5] = 1
    Y3[0][N - 5] = 1

def getImage4():
    for i in range(N):
        for j in range(N):
            Y4[i][j] = 0
    Y4[5][0] = 1
    Y4[N - 5][0] = 1

def getImage5():
    for i in range(N):
        for j in range(N):
            Y5[i][j] = 0
    Y5[5][5] = 1
    Y5[N - 5][N - 5] = 1

N = 50
Y1 = np.empty((N, N))
Y2 = np.empty((N, N))
Y3 = np.empty((N, N))
Y4 = np.empty((N, N))
Y5 = np.empty((N, N))

time = np.random.randint(0, 256, (N, N, 3), dtype=np.uint8)

# imaginea 1
getImage1()

# imagine
plt.imshow(Y1, cmap=plt.cm.gray)
plt.show()

plt.savefig('Ex1_1_1.png', format='png')
plt.savefig('Ex1_1_1.pdf', format='pdf')

# spectograma
Y1 = np.fft.fft2(Y1)
freq_db1 = 20 * np.log10(abs(Y1))
plt.imshow(freq_db1)
plt.colorbar()
plt.show()

plt.savefig('Ex1_1_2.png', format='png')
plt.savefig('Ex1_1_2.pdf', format='pdf')

# imaginea 2
getImage2()

# imagine
plt.imshow(Y2, cmap=plt.cm.gray)
plt.show()

plt.savefig('Ex1_2_1.png', format='png')
plt.savefig('Ex1_2_1.pdf', format='pdf')

# spectograma
Y2 = np.fft.fft2(Y2)
freq_db2 = 20 * np.log10(abs(Y2))
plt.imshow(freq_db2)
plt.colorbar()
plt.show()

plt.savefig('Ex1_2_2.png', format='png')
plt.savefig('Ex1_2_2.pdf', format='pdf')

# imaginea 3
getImage3()

# imagine
plt.imshow(Y3, cmap=plt.cm.gray)
plt.show()

plt.savefig('Ex1_3_1.png', format='png')
plt.savefig('Ex1_3_1.pdf', format='pdf')

# spectograma
Y3 = np.fft.fft2(Y3)
freq_db3 = 20 * np.log10(abs(Y3))
plt.imshow(freq_db3)
plt.colorbar()
plt.show()

plt.savefig('Ex1_3_2.png', format='png')
plt.savefig('Ex1_3_2.pdf', format='pdf')

# imaginea 4
getImage4()

# imagine
plt.imshow(Y4, cmap=plt.cm.gray)
plt.show()

plt.savefig('Ex1_4_1.png', format='png')
plt.savefig('Ex1_4_1.pdf', format='pdf')

# spectograma
Y4 = np.fft.fft2(Y4)
freq_db4 = 20 * np.log10(abs(Y4))
plt.imshow(freq_db4)
plt.colorbar()
plt.show()

plt.savefig('Ex1_4_2.png', format='png')
plt.savefig('Ex1_4_2.pdf', format='pdf')

# imaginea 5
getImage5()

# imagine
plt.imshow(Y5, cmap=plt.cm.gray)
plt.show()

plt.savefig('Ex1_5_1.png', format='png')
plt.savefig('Ex1_5_1.pdf', format='pdf')

# spectograma
Y5 = np.fft.fft2(Y5)
freq_db5 = 20 * np.log10(abs(Y5))
plt.imshow(freq_db5)
plt.colorbar()
plt.show()

plt.savefig('Ex1_5_2.png', format='png')
plt.savefig('Ex1_5_2.pdf', format='pdf')
