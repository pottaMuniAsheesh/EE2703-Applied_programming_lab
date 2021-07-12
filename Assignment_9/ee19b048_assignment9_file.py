import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

fignum = 1

plt.rcParams.update({'text.usetex': True})

def ex1():
    t = np.linspace(0, 2*np.pi, 129)[:-1]
    x = np.sin(5*t)
    Y = fft(x)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.abs(Y))
    plt.ylabel(r'$|X|$', size=16)
    plt.title(r'Spectrum of $x(t) = sin(5t)$')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(np.unwrap(np.angle(Y)))
    plt.ylabel(r'$\angle X$', size=16)
    plt.xlabel(r'$k$', size=16)
    plt.grid(True)

def q1():
    N = 512
    T = (8*np.pi)/N
    t = np.linspace(0, N*T, N+1)
    t = t[:-1]
    x = np.sin(5*t)
    x_spec = np.fft.fftshift(np.fft.fft(x))/N
    y = (1 + 0.1*np.cos(t))*np.cos(10*t)
    y_spec = np.fft.fftshift(np.fft.fft(y))/N
    W = 2*np.pi/T
    w = np.linspace(-0.5*W, 0.5*W, N+1)
    w = w[:-1]
    global fignum 
    plt.figure(fignum)
    fignum += 1
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(x_spec))
    plt.xlim([-15,15])
    plt.grid(True)
    plt.ylabel(r'$|X|$', size=16)
    plt.title(r'Spectrum of $x(t) = sin(5t)$')
    plt.subplot(2,1,2)
    ii = np.where(np.abs(x_spec) > 1e-3)
    plt.plot(w[ii], np.angle(x_spec)[ii], 'ro')
    plt.ylabel(r'$\angle X$', size=16)
    plt.xlabel(r'$\omega$', size=16)
    plt.xlim([-15,15])
    plt.ylim((-2,2))
    plt.grid(True)
    plt.tight_layout()

    plt.figure(fignum)
    fignum += 1
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(y_spec))
    plt.xlim([-15,15])
    plt.grid(True)
    plt.ylabel(r'$|Y|$', size=16)
    plt.title(r'Spectrum of $y(t) = (1 + 0.1cos(t))cos(10t)$')
    plt.subplot(2,1,2)
    ii = np.where(np.abs(y_spec) > 1e-3)
    plt.plot(w[ii], np.angle(y_spec)[ii], 'ro')
    plt.ylabel(r'$\angle Y$', size=16)
    plt.xlabel(r'$\omega$', size=16)
    plt.xlim([-15,15])
    plt.ylim((-2,2))
    plt.grid(True)
    plt.tight_layout()

def q2():
    N = 512
    T = (8*np.pi)/N
    t = np.linspace(0, 8*np.pi, N+1)
    t = t[:-1]
    x = np.sin(t)**3
    x_spec = np.fft.fftshift(np.fft.fft(x))/N
    y = np.cos(t)**3
    y_spec = np.fft.fftshift(np.fft.fft(y))/N
    W = 2*np.pi/T
    w = np.linspace(-0.5*W, 0.5*W, N+1)
    w = w[:-1]
    global fignum 
    plt.figure(fignum)
    fignum += 1
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(x_spec))
    plt.xlim([-15,15])
    plt.grid(True)
    plt.ylabel(r'$|X|$', size=16)
    plt.title(r'Spectrum of $x(t) = sin^3(t)$')
    plt.subplot(2,1,2)
    ii = np.where(np.abs(x_spec) > 1e-3)
    plt.plot(w[ii], np.angle(x_spec)[ii], 'ro')
    plt.ylabel(r'$\angle X$', size=16)
    plt.xlabel(r'$\omega$', size=16)
    plt.xlim([-15,15])
    plt.ylim((-2,2))
    plt.grid(True)
    plt.tight_layout()

    plt.figure(fignum)
    fignum += 1
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(y_spec))
    plt.xlim([-15,15])
    plt.grid(True)
    plt.ylabel(r'$|Y|$', size=16)
    plt.title(r'Spectrum of $y(t) = cos^3(t)$')
    plt.subplot(2,1,2)
    ii = np.where(np.abs(y_spec) > 1e-3)
    plt.plot(w[ii], np.angle(y_spec)[ii], 'ro')
    plt.ylabel(r'$\angle Y$', size=16)
    plt.xlabel(r'$\omega$', size=16)
    plt.xlim([-15,15])
    plt.ylim((-2,2))
    plt.grid(True)
    plt.tight_layout()

def q3():
    N = 512
    T = (8*np.pi)/N
    t = np.linspace(0, 8*np.pi, N+1)
    t = t[:-1]
    x = np.cos(20*t + 5*np.cos(t))
    x_spec = np.fft.fftshift(np.fft.fft(x))/N
    W = 2*np.pi/T
    w = np.linspace(-0.5*W, 0.5*W, N+1)
    w = w[:-1]
    global fignum 
    plt.figure(fignum)
    fignum += 1
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(x_spec))
    plt.xlim([-40,40])
    plt.grid(True)
    plt.ylabel(r'$|X|$', size=16)
    plt.title(r'Spectrum of $x(t) = cos(20t + 5cos(t))$')
    plt.subplot(2,1,2)
    ii = np.where(np.abs(x_spec) > 1e-3)
    plt.plot(w[ii], np.angle(x_spec)[ii], 'ro')
    plt.ylabel(r'$\angle X$', size=16)
    plt.xlabel(r'$\omega$', size=16)
    plt.xlim([-40,40])
    # plt.ylim((-2,2))
    plt.grid(True)
    plt.tight_layout()

def q4():
    N = 256 # number of samples
    T = (4*np.pi)/N # sampling time interval
    t = np.linspace(-0.5*N*T, 0.5*N*T, N+1)
    t = t[:-1]
    x = np.exp(-0.5*t*t)
    x_spec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))*(T/(2*np.pi))
    W = 2*np.pi/T
    w = np.linspace(-0.5*W, 0.5*W, N+1)
    w = w[:-1]
    global fignum

    # plt.figure(fignum)
    # fignum += 1
    actual_x_spec = (1/np.sqrt(2*np.pi))*np.exp(-0.5*w*w)
    # plt.plot(w, actual_x_spec)
    # plt.xlim((-10,10))
    # plt.title(r"$X(\omega) = 1/\sqrt{2\pi}exp(-\omega^2/2)$")
    # plt.grid(True)

    plt.figure(fignum)
    fignum += 1
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(x_spec),'ro', label=r'Computed spectrum')
    plt.plot(w, np.abs(actual_x_spec),'b', label=r'Actual spectrum')
    plt.xlim([-10,10])
    plt.grid(True)
    plt.ylabel(r'$|X|$', size=16)
    plt.title(r'Spectrum of $x(t) = exp(-t^2/2)$')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(w, np.angle(x_spec), 'ro', label=r'Computed spectrum')
    plt.plot(w, np.angle(actual_x_spec), 'b', label=r'Actual spectrum')
    plt.ylabel(r'$\angle X$', size=16)
    plt.xlabel(r'$\omega$', size=16)
    plt.xlim([-10,10])
    # plt.ylim((-2,2))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    max_abs_error = np.max(np.abs(x_spec - actual_x_spec))
    print("The maximum absolute error in the spectrum is {}".format(max_abs_error))

# ex1()
# q1()
# q2()
# q3()
q4()

plt.show()