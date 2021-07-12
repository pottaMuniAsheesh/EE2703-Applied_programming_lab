import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys

# Default values
n = 100  # number of sections the tubelight is divided into
M = 10   # mean of number of electrons injected per turn
Msig = 2 # std-dev of number of electrons injected per turn
nk = 500 # number of turns to simulate
u0 = 7   # threshold velocity
p = 0.5  # probability of ionization

# Taking custom values 
if len(sys.argv[1:]) == 6:
    n = int(sys.argv[1])
    M, Msig, nk, u0, p = map(float, sys.argv[2:])

xx = np.zeros(n*M) # electron position
u = np.zeros(n*M)  # electron velocity
dx = np.zeros(n*M) # electron displacement in present turn

I = []
X = []
V = []

for _ in range(nk):

    existing = np.where(xx>0)[0]
    X.extend(xx[existing].tolist())
    V.extend(u[existing].tolist())
    dx[existing] = u[existing] + 0.5
    xx[existing] += dx[existing]
    u[existing] += 1

    absorbed = np.where(xx > n)[0]
    dx[absorbed] = 0
    u[absorbed] = 0
    xx[absorbed] = 0

    energetic = np.where(u >= u0)[0]
    jj = np.where(np.random.rand(energetic.size) <= p)[0]
    collided = energetic[jj]
    u[collided] = 0
    rho = np.random.rand(collided.size)
    xx[collided] = xx[collided] - dx[collided]*rho

    I.extend(xx[collided].tolist())

    m = int(np.random.randn()*Msig + M)
    empty = np.where(xx == 0)[0]
    m = int(min(m, empty.size))
    xx[empty[:m]] = 1

plt.figure(1)
plt.hist(X, 100, color='blue', range=(0,100), density=True)
plt.title('Electron density')

plt.figure(2)
count, bins = plt.hist(I, 100, color='blue', range=(0,100), density=True)[:2]
plt.title('Emission Intensity')

plt.figure(3)
plt.plot(X, V, 'ro')
plt.title('Phase space')

xpos = 0.5*(bins[:-1]+bins[1:])
print('Intensity data:')
print(tabulate(np.c_[xpos,count], headers=['xpos','count']))

# plt.show()