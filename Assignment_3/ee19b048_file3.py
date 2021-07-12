import numpy as np
from scipy.special import jv
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

def g(t, A, B):
    return ((A*jv(2,t)) + (B*t))

data = np.loadtxt('fitting.dat',unpack=True)
time = data[0]
true_value = g(time,1.05,-0.105)
M = np.c_[jv(2,time),time]
A0, B0 = 1.05, -0.105
noise_std_dev = np.logspace(-1,-3,9)

# Q4
f1 = plt.figure(1)
for i in range(1,10):
    plt.plot(time, data[i], label=r'$\sigma_%d=%.3f$'%(i,noise_std_dev[i-1]))
plt.plot(time, true_value, 'k-', lw=2, label='true value')
plt.title('Data to be fitted to theory')
plt.ylabel('f(t)+noise'+r'$\longrightarrow$')
plt.xlabel('t'+r'$\longrightarrow$')
plt.legend()

# Q5
f2 = plt.figure(2)
plt.errorbar(time[::5],data[1][::5],0.1,fmt='ro', label='Errorbar')
plt.plot(time,true_value,label='true value')
plt.legend()
plt.xlabel(r't $\longrightarrow$')
plt.title(r'Q5: Data points for $\sigma$ = 0.10 along with exact function')
plt.grid(True)

# Q6
f3 = plt.figure(3)
plt.plot(time, np.dot(M,np.array([[A0],[B0]])), lw=3, label='M.p')
plt.plot(time, g(time, A0, B0), label='g(t, A0, B0)')
plt.xlabel(r't $\longrightarrow$')
plt.legend()

# Q7, Q8
A = np.linspace(0, 2, 21)
B = np.linspace(-0.2, 0, 21)
mean_squared_error = np.zeros((A.size, B.size))
for i in range(A.size):
    for j in range(B.size):
        mean_squared_error[i,j] += (1/101)*(np.sum(np.square(data[1]-g(time,A[i],B[j]))))
min_error_i, min_error_j = np.unravel_index(mean_squared_error.argmin(), mean_squared_error.shape)
f4, ax = plt.subplots()
CS = ax.contour(A, B, mean_squared_error, np.arange(0,1,0.025))
ax.clabel(CS, CS.levels[:5], inline=1, fontsize=10)
plt.plot(A[min_error_i], B[min_error_j], 'ro', label='min')
plt.plot(1.05, -0.105, 'go', label='Exact location')
plt.annotate('(%.2f,%.2f)'%(A[min_error_i],B[min_error_j]), (A[min_error_i],B[min_error_j]))
plt.annotate('(1.05,-0.105)', (1.05,-0.105))
plt.xlabel(r'$A$ $\longrightarrow$')
plt.ylabel(r'$B$ $\longrightarrow$')
plt.title(r'Q8: Contour plot of $\epsilon_{ij}$')
plt.grid(True)
plt.legend()

# Q9, Q10
Aerr = np.zeros(9)
Berr = np.zeros(9)
for i in range(1,10):
    x = data[i]
    p, *residue = lstsq(M, x)
    Aerr[i-1] = np.abs(1.05-p[0])
    Berr[i-1] = np.abs(-0.105-p[1])
f5 = plt.figure(5)
plt.plot(noise_std_dev, Aerr, 'ro:', label='Aerr')
plt.plot(noise_std_dev, Berr, 'go:', label='Berr')
plt.xlabel(r'$\sigma_n$ $\longrightarrow$')
plt.ylabel(r'$Error$ $\longrightarrow$')
plt.title('Q10: Variation of error with noise')
plt.grid(True)
plt.legend()

# Q11
f6 = plt.figure(6)
plt.stem(noise_std_dev, Aerr, 'r-', 'ro', label='Aerr')
plt.stem(noise_std_dev, Berr, 'g-', 'go', label='Berr')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\sigma_n$ $\longrightarrow$')
plt.ylabel(r'$Error$ $\longrightarrow$')
plt.title('Q11: Variation of error with noise')
plt.legend()

plt.show()