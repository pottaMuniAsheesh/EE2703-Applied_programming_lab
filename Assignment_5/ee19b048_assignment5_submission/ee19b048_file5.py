import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.linalg import lstsq

# default values for arguments
Nx = 25
Ny = 25
radius = 8
Niter = 1500

# taking commandline args
if len(sys.argv) == 5:
    Nx, Ny, radius, Niter = sys.argv[1:]
    Nx = int(Nx)
    Ny = int(Ny)
    Niter = int(Niter)
    radius = float(radius)
    if not (Niter > 500):
        print('Number of iterations should be greater than 500')
        exit()
    if radius > Nx//2:
        print('Radius greater than plate size')
        exit()
    if Nx != Ny:
        print('Nx and Ny should be same')
        exit()

phi = np.zeros((Ny, Nx))
y = np.linspace(0.5*(Ny-1), -0.5*(Ny-1), Ny)
x = np.linspace(-0.5*(Nx-1), 0.5*(Nx-1), Nx)
X, Y = np.meshgrid(x,y)
ii = np.where(((X)*(X) + (Y)*(Y)) <= radius*radius)
phi[ii] = 1.0
errors = np.zeros(Niter)

for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1,1:-1] = 0.25*(phi[1:-1,:-2]+phi[1:-1,2:]+phi[:-2,1:-1]+phi[2:,1:-1])
    phi[1:-1,0] = phi[1:-1,1]
    phi[1:-1,-1] = phi[1:-1,-2]
    phi[0,:] = phi[1,:]
    phi[ii] = 1.0
    errors[k] = np.max(np.abs(phi-oldphi))

plt.figure(1)
plt.semilogy(np.arange(0,Niter,50), errors[::50], 'ro', label='Errors')
plt.grid(True)

M = np.zeros((Niter,2))
M[:,0] = 1
M[:,1] = np.arange(Niter)
fit1_params = lstsq(M, np.log(errors))[0]
fit2_params = lstsq(M[500:,:], np.log(errors[500:]))[0]
fit1_errors = np.exp(np.dot(M, fit1_params))
fit2_errors = np.exp(np.dot(M, fit2_params))
fit1_params[0] = np.exp(fit1_params[0])
fit2_params[0] = np.exp(fit2_params[0])

# print("For fit using all errors: A = {}, B = {}".format(*fit1_params))
# print("For fit using errors after 500 iterations: A = {}, B = {}".format(*fit2_params))

plt.figure(1)
plt.semilogy(np.arange(0,Niter,50), fit1_errors[::50], 'g-', label='Fit using all errors')
plt.semilogy(np.arange(0,Niter,50), fit2_errors[::50], 'b-', label='Fit using errors after 500 iterations')
plt.title('Error with each iteration')
plt.xlabel(r'Error$\longrightarrow$')
plt.ylabel(r'Number of iterations$\longrightarrow$')
plt.legend()

fig2 = plt.figure(2)
ax = p3.Axes3D(fig2)
s = ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlabel(r'X$\longrightarrow$')
ax.set_ylabel(r'Y$\longrightarrow$')
ax.set_title('3-D surface plot of potential')
plt.colorbar(s, ax = ax, shrink=0.7)

plt.figure(3)
plt.contourf(X, Y, phi, (Nx+Ny)//4, cmap=cm.jet)
plt.plot(X[ii], Y[ii], 'ro')
plt.title(r'Potential $\phi$')
plt.xlabel(r'X$\longrightarrow$')
plt.ylabel(r'Y$\longrightarrow$')
plt.colorbar()

Jx = np.zeros((Ny, Nx))
Jy = np.zeros((Ny, Nx))
Jx[1:-1,1:-1] = 0.5*(phi[1:-1,:-2]-phi[1:-1,2:])
Jy[1:-1,1:-1] = 0.5*(phi[2:,1:-1]-phi[:-2,1:-1])
plt.figure(4)
plt.quiver(X, Y, Jx, Jy, scale=5)
plt.plot(X[ii], Y[ii], 'ro')
plt.xlabel(r'X$\longrightarrow$')
plt.ylabel(r'Y$\longrightarrow$')
plt.title('The vector plot of current flow')

# Solving the temperature equation
T = np.zeros((Ny, Nx))
T[:,:] = 300.0
del_x = 0.005/Nx # Assuming Nx == Ny
mag_sq_J = (Jx*Jx + Jy*Jy)
for k in range(Niter):
    # oldT = T.copy()
    T[1:-1,1:-1] = 0.25*(T[1:-1,:-2]+T[1:-1,2:]+T[:-2,1:-1]+T[2:,1:-1] + mag_sq_J[1:-1, 1:-1])
    T[1:-1,0] = T[1:-1,1]
    T[1:-1,-1] = T[1:-1,-2]
    T[0,:] = T[1,:]
    T[ii] = 300.0
plt.figure(5)
plt.contourf(X, Y, T, Nx//2, cmap = cm.jet)
plt.title('Temperature of plate')
plt.xlabel(r'X$\longrightarrow$')
plt.ylabel(r'Y$\longrightarrow$')
plt.colorbar()

plt.show()