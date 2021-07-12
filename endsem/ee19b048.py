import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

# Using Latex in plots.
plt.rcParams.update({'text.usetex':True})

a = 10; k = 1/a

def calc(l, x1, y1, dx1, dy1, X, Y, Z, phi, k, dynamic=True):
    """
    This function finds the vector potential due to the current
    element of index l. Here, the current I = (4*pi/mu0)*cos(phi),
    and depending on whether the current is time-dependent or not
    (determined by the boolean parameter 'dynamic'), the current
    is multiplied by exp(j*w*t). Here, first, R = |r - r'| is 
    found and then vector potential A is computed, which has 
    only x and y components.
    """
    Rl = np.sqrt((X-x1[l])**2 + (Y-y1[l])**2 + Z**2)
    if dynamic:
        A_xl = (np.cos(phi[l])*np.exp(-1j*k*Rl)*dx1[l])/Rl
        A_yl = (np.cos(phi[l])*np.exp(-1j*k*Rl)*dy1[l])/Rl
    else:
        A_xl = (np.cos(phi[l])*dx1[l])/Rl
        A_yl = (np.cos(phi[l])*dy1[l])/Rl
    return np.array([A_xl, A_yl])

def calc1(l, x1, y1, dx1, dy1, X, Y, Z, phi, k, dynamic=True):
    """
    This function finds the vector potential due to the current
    element of index l. Here, the current I = (4*pi/mu0)*1,
    and depending on whether the current is time-dependent or not
    (determined by the boolean parameter 'dynamic'), the current
    is multiplied by exp(j*w*t). Here, first, R = |r - r'| is 
    found and then vector potential A is computed, which has 
    only x and y components.
    """
    Rl = np.sqrt((X-x1[l])**2 + (Y-y1[l])**2 + Z**2)
    if dynamic:
        A_xl = (1*np.exp(-1j*k*Rl)*dx1[l])/Rl
        A_yl = (1*np.exp(-1j*k*Rl)*dy1[l])/Rl
    else:
        A_xl = (1*dx1[l])/Rl
        A_yl = (1*dy1[l])/Rl
    return np.array([A_xl, A_yl])

x = np.linspace(-1.0,1.0,3)  
y = np.linspace(-1.0,1.0,3)
z = np.linspace(1.0, 1000.0, 1000)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
phi = np.linspace(0, 2*np.pi, 101); phi = phi[:-1]
delta_phi = phi[1]-phi[0]
phi += (phi[1]-phi[0])/2 # Setting the point to middle of the element.

x1, y1 = a*np.cos(phi), a*np.sin(phi) # r' vector components
dx1, dy1 = -a*delta_phi*np.sin(phi), a*delta_phi*np.cos(phi) # dl' vector components
# x and y components of the current element vector (Idl'), 
# alternating elements are considered for visual comfort.
I_x, I_y = (x1[::2]/a)*(dx1[::2]), (x1[::2]/a)*(dy1[::2])

# plotting the vector arrows of current elements in the loop.
fig1 = plt.figure(1, figsize=(8,8))
ax = fig1.add_subplot(111)
ax.plot(x1, y1, 'r.', label='current element')
ax.quiver(x1[::2], y1[::2], I_x, I_y, label=r"$I\vec{dl'}$")
plt.grid(True)
plt.legend(loc=1, fontsize='large')
plt.title('Current elements', size=16)
plt.xlabel('x (in cm)')
plt.ylabel('y (in cm)')

# Since summation over a single dimension is not possible with vectorized code,
# for loop needs to be used.
A = calc(0, x1, y1, dx1, dy1, X, Y, Z, phi, k, dynamic=True)
for l in range(1, 100):
    A += calc(l, x1, y1, dx1, dy1, X, Y, Z, phi, k, dynamic=True)

# Un-commment this section and comment the above part, to find magentic field
# Bz for constant current w.r.t phi, i.e; I = 4pi/mu0. 
# A = calc1(0, x1, y1, dx1, dy1, X, Y, Z, phi, k, dynamic=True)
# for l in range(1, 100):
#     A += calc1(l, x1, y1, dx1, dy1, X, Y, Z, phi, k, dynamic=True)

# Finding the curl of the vector potential along the z-axis to find the magentic field.
B_z = np.abs(0.5*(A[1,2,1,:]-A[0,1,2,:]-A[1,0,1,:]+A[0,1,0,:]))

plt.figure(2)
plt.loglog(z, B_z, label=r'$B_z(z)$')
plt.title(r'Plot of z-component magnetic field $B_z(z)$')
plt.xlabel(r'$z$ (in cm)', size=12)
plt.ylabel(r'$B_z(z)$', size=12)
plt.grid(True)

# Finding the least squares fit (b, c) for the model, Bz = c*(z^b)
M = np.c_[np.log(z),np.ones(z.size)]
fit = lstsq(M, np.log(B_z))[0]
b = fit[0]
c = np.exp(fit[1])
print("The approximated values of b and c are {}, {}".format(b, c))
plt.figure(3)
plt.loglog(z, B_z, label=r'$B_z(z)$')
plt.title(r'Plot of z-component magnetic field $B_z(z)$ and Least squares fit')
plt.xlabel(r'$z$ (in cm)', size=12)
plt.ylabel(r'$B_z(z)$', size=12)
plt.grid(True)
plt.loglog(z, np.exp(np.dot(M, fit)), 'ro', label=r'Least sqaures fit')
plt.legend()

plt.show()

# print(X, Y, Z, sep='\n')