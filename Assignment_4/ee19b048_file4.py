import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import lstsq

# defining the required functions
def e(x):
    return np.exp(x)

def coscos(x):
    return np.cos(np.cos(x))

def u(x, k, f):
    return f(x)*np.cos(k*x)

def v(x, k, f):
    return f(x)*np.sin(k*x)

# considering 1200 x values from -2pi to 4pi
x = np.linspace(-2*pi, 4*pi, 1201)
x = x[:-1]

fig1 = plt.figure(1)
plt.semilogy(x, e(x), 'r-', label=r'$exp(x)$')
plt.xlabel(r'$x\longrightarrow$')
plt.ylabel(r'$exp(x)\longrightarrow$')
plt.grid(True)
plt.title('Plot of exp(x)')
plt.legend()

fig2 = plt.figure(2)
plt.plot(x, coscos(x), 'r-', label=r'$cos(cos(x))$')
plt.xlabel(r'$x\longrightarrow$')
plt.ylabel(r'$cos(cos(x))\longrightarrow$')
plt.grid(True)
plt.title('Plot of cos(cos(x))')
plt.legend()

# initializing coefficient vectors
coeffs_f1 = np.zeros(51)
coeffs_f2 = np.zeros(51)
# computing the coefficients by integration
coeffs_f1[0] = quad(u, 0, 2*pi, args=(0,e))[0]/(2*pi)
coeffs_f2[0] = quad(u, 0, 2*pi, args=(0,coscos))[0]/(2*pi)

for i in range(1, 26):
    coeffs_f1[2*i-1] = quad(u, 0, 2*pi, args=(i,e))[0]/(pi)
    coeffs_f1[2*i] = quad(v, 0, 2*pi, args=(i,e))[0]/(pi)
    coeffs_f2[2*i-1] = quad(u, 0, 2*pi, args=(i,coscos))[0]/(pi)
    coeffs_f2[2*i] = quad(v, 0, 2*pi, args=(i,coscos))[0]/(pi)

plt.figure(3)
n = np.linspace(1,51,51)
plt.semilogy(n, np.abs(coeffs_f1), 'ro', label='Direct Integration')
plt.xlabel(r'n$\longrightarrow$')
plt.title(r'Fourier coefficients of $exp(x)$')
plt.grid(True)

plt.figure(4)
plt.loglog(n, np.abs(coeffs_f1), 'ro', label='Direct Integration')
plt.xlabel(r'n$\longrightarrow$')
plt.title(r'Fourier coefficients of $exp(x)$')
plt.grid(True)

plt.figure(5)
plt.semilogy(n, np.abs(coeffs_f2), 'ro', label='Direct Integration')
plt.xlabel(r'n$\longrightarrow$')
plt.title(r'Fourier coefficients of $cos(cos(x))$')
plt.grid(True)

plt.figure(6)
plt.loglog(n, np.abs(coeffs_f2), 'ro', label='Direct Integration')
plt.xlabel(r'n$\longrightarrow$')
plt.title(r'Fourier coefficients of $cos(cos(x))$')
plt.grid(True)

# finding the least squares approximation of coeffcients
x = np.linspace(0, 2*pi, 401)
x = x[:-1]
b1 = e(x)
b2 = coscos(x)
A = np.zeros((400, 51))
A[:,0] = 1
for k in range(1, 26):
    A[:,2*k-1] = np.cos(k*x)
    A[:,2*k] = np.sin(k*x)
c1 = lstsq(A, b1)[0]
c2 = lstsq(A, b2)[0]

plt.figure(3)
plt.semilogy(n, np.abs(c1), 'go', label='Least Squares')
plt.legend()

plt.figure(4)
plt.loglog(n, np.abs(c1), 'go', label='Least Squares')
plt.legend()

plt.figure(5)
plt.semilogy(n, np.abs(c2), 'go', label='Least Squares')
plt.legend()

plt.figure(6)
plt.loglog(n, np.abs(c2), 'go', label='Least Squares')
plt.legend()

# finding the maximum deviation of approximated coefficients from coefficients obtained by integration
dev_f1 = np.max(np.abs(c1 - coeffs_f1))
dev_f2 = np.max(np.abs(c2 - coeffs_f2))

# finding the function values using approximated coefficients
Ac1 = np.dot(A, c1)
Ac2 = np.dot(A, c2)

plt.figure(1)
plt.semilogy(x, Ac1, 'go', label='Least Squares')
plt.legend()

plt.figure(2)
plt.plot(x, Ac2, 'go', label='Least Squares')
plt.legend()

plt.figure(7)
plt.semilogy(x, e(x), 'r-')
plt.grid(True)
plt.xlabel(r'$x\longrightarrow$')
plt.ylabel(r'$exp(x)\longrightarrow$')
plt.title(r'Plot of $exp(x)$ over $[0,2\pi)$')

plt.figure(8)
plt.plot(x, coscos(x), 'r-')
plt.grid(True)
plt.xlabel(r'$x\longrightarrow$')
plt.ylabel(r'$cos(cos(x))\longrightarrow$')
plt.title(r'Plot of $cos(cos(x))$ over $[0,2\pi)$')

plt.show()

print(dev_f1, dev_f2)