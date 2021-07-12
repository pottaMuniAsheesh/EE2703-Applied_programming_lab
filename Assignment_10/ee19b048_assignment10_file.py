import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3

plt.rcParams.update({'text.usetex': True})

def ex1():
    t=np.linspace(-np.pi,np.pi,65);t=t[:-1]
    dt=t[1]-t[0];fmax=1/dt
    y=np.sin(np.sqrt(2)*t)
    y[0]=0 # the sample corresponding to -tmax should be set zero
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/64.0
    w=np.linspace(-np.pi*fmax,np.pi*fmax,65);w=w[:-1]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w,abs(Y),lw=2)
    plt.xlim([-10,10])
    plt.ylabel(r"$|Y|$",size=16)
    plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(w,np.angle(Y),'ro',lw=2)
    plt.xlim([-10,10])
    plt.ylabel(r"$\angle Y$",size=16)
    plt.xlabel(r"$\omega$",size=16)
    plt.grid(True)

def ex2():
    t1 = np.linspace(-np.pi, np.pi, 65); t1 = t1[:-1]
    t2 = np.linspace(-3*np.pi, -np.pi, 65); t2 = t2[:-1]
    t3 = np.linspace(np.pi, 3*np.pi, 65); t3 = t3[:-1]
    plt.figure()
    plt.plot(t1, np.sin(np.sqrt(2)*t1), 'b', lw=2)
    plt.plot(t2, np.sin(np.sqrt(2)*t2), 'r', lw=2)
    plt.plot(t3, np.sin(np.sqrt(2)*t3), 'r', lw=2)
    plt.title(r'$sin(\sqrt{2}t)$')
    plt.xlabel(r'$t$', size=12)
    plt.grid(True)

def ex3():
    t1 = np.linspace(-np.pi, np.pi, 65); t1 = t1[:-1]
    t2 = np.linspace(-3*np.pi, -np.pi, 65); t2 = t2[:-1]
    t3 = np.linspace(np.pi, 3*np.pi, 65); t3 = t3[:-1]
    y = np.sin(np.sqrt(2)*t1)
    plt.figure()
    plt.plot(t1, y, 'b', lw=2)
    plt.plot(t2, y, 'r', lw=2)
    plt.plot(t3, y, 'r', lw=2)
    plt.title(r'$sin(\sqrt{2}t)$ with wrapping $t$ every $2\pi$ interval')
    plt.xlabel(r'$t$')
    plt.grid(True)

def ex4():
    t = np.linspace(-np.pi, np.pi, 65); t = t[:-1]
    t2 = np.linspace(-3*np.pi, -np.pi, 65); t2 = t2[:-1]
    t3 = np.linspace(np.pi, 3*np.pi, 65); t3 = t3[:-1]
    dt = t[1]-t[0]
    fmax = 1/dt
    y = t

    plt.figure()
    plt.plot(t2, y, 'b')
    plt.plot(t, y, 'b')
    plt.plot(t3, y, 'b')
    plt.title(r'Periodic ramp signal')
    plt.xlabel(r'time $t$')
    plt.grid(True)

    y[0] = 0
    y = fftshift(y)
    y_spec = fftshift(fft(y))/64.0
    w = np.linspace(-np.pi*fmax, np.pi*fmax, 65); w = w[:-1]

    plt.figure()
    plt.semilogx(np.abs(w), 20*np.log10(np.abs(y_spec)))
    plt.xlim([1, 10])
    plt.ylim([-20, 0])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$|Y|$ (in dB)')
    plt.title(r'Magnitude spectrum of $y(t) = sin(\sqrt{2}t)$')
    plt.grid(True)

def hamming_window(N):
    n = np.arange(N)
    return fftshift(0.54 + 0.46*np.cos(2*np.pi*n/(N-1)))

def ex5():
    t1 = np.linspace(-np.pi, np.pi, 65); t1 = t1[:-1]
    t2 = np.linspace(-3*np.pi, -np.pi, 65); t2 = t2[:-1]
    t3 = np.linspace(np.pi, 3*np.pi, 65); t3 = t3[:-1]
    y = np.sin(np.sqrt(2)*t1)*hamming_window(64)
    plt.figure()
    plt.plot(t1, y, 'b', lw=2)
    plt.plot(t2, y, 'r', lw=2)
    plt.plot(t3, y, 'r', lw=2)
    plt.title(r'$sin(\sqrt{2}t) \times w(t)$ with wrapping $t$ every $2\pi$ interval')
    plt.xlabel(r'$t$')
    plt.grid(True)

def ex6():
    t=np.linspace(-np.pi,np.pi,65);t=t[:-1]
    dt=t[1]-t[0];fmax=1/dt
    y=np.sin(np.sqrt(2)*t)*hamming_window(64)
    y[0]=0 # the sample corresponding to -tmax should be set zero
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/64.0
    w=np.linspace(-np.pi*fmax,np.pi*fmax,65);w=w[:-1]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w,abs(Y),lw=2)
    plt.xlim([-10,10])
    plt.ylabel(r"$|Y|$",size=16)
    plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right) \times w(t)$")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(w,np.angle(Y),'ro',lw=2)
    plt.xlim([-10,10])
    plt.ylabel(r"Phase of $Y$",size=16)
    plt.xlabel(r"$\omega$",size=16)
    plt.grid(True)

def ex7():
    t=np.linspace(-4*np.pi,4*np.pi,257);t=t[:-1]
    dt=t[1]-t[0];fmax=1/dt
    # y=np.sin(np.sqrt(2)*t)
    y = np.sin(1.5*t)
    y = y*hamming_window(256)
    y[0]=0 # the sample corresponding to -tmax should be set zero
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/256.0
    w=np.linspace(-np.pi*fmax,np.pi*fmax,257);w=w[:-1]
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w,abs(Y),'b', lw=2)
    plt.plot(w,abs(Y),'bo', lw=2)
    plt.xlim([-4,4])
    plt.ylabel(r"$|Y|$",size=16)
    # plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right) \times w(t)$")
    plt.title(r"Spectrum of $\sin\left(1.5t\right) \times w(t)$")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(w,np.angle(Y),'ro',lw=2)
    plt.xlim([-4,4])
    plt.ylabel(r"Phase of $Y$",size=16)
    plt.xlabel(r"$\omega$",size=16)
    plt.grid(True)

def q2():
    N = 256
    t = np.linspace(-(N/64)*np.pi, (N/64)*np.pi, N+1); t = t[:-1]
    dt = t[1] - t[0]
    w0 = 0.86
    y1 = np.cos(w0*t)**3
    y2 = y1*hamming_window(N)
    w = np.linspace(-np.pi/dt, np.pi/dt, N+1); w = w[:-1]
    y1[0] = 0; y2[0] = 0
    y1 = fftshift(y1); y2 = fftshift(y2)
    y1_spec = fftshift(fft(y1))/N; y2_spec = fftshift(fft(y2))/N

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(y1_spec), 'b', lw=2)
    plt.xlim([-5,5])
    plt.ylabel(r'$|Y|$', size=12)
    plt.title(r'Spectrum of $cos^3(0.86t)$')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(w, np.angle(y1_spec), 'ro')
    plt.xlim([-5,5])
    plt.xlabel(r'$\omega$', size=12)
    plt.ylabel(r'$\angle Y$', size=12)
    plt.grid(True)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(y2_spec), 'b', lw=2)
    plt.xlim([-5,5])
    plt.ylabel(r'$|Y|$', size=12)
    plt.title(r'Spectrum of $cos^3(0.86t) \times w(t)$')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(w, np.angle(y2_spec), 'ro')
    plt.xlim([-5,5])
    plt.xlabel(r'$\omega$', size=12)
    plt.ylabel(r'$\angle Y$', size=12)
    plt.grid(True)

def arbit_cos(noise=False):
    t = np.linspace(-np.pi, np.pi, 129); t = t[:-1]
    w0 = np.random.rand() + 0.5
    delta = np.random.rand()*2*np.pi
    # if delta-np.pi > 0:
    #     delta = 2*np.pi - delta
    if noise:
        return t, np.cos(w0*t + delta) + 0.1*np.random.randn(128), w0, delta
    return t, np.cos(w0*t + delta), w0, delta

def q3():
    t, y, w0, delta = arbit_cos()
    dt = t[1] - t[0]
    y[0] = 0
    y = fftshift(y)
    y_spec = fftshift(fft(y))/128
    y_spec_mag = np.abs(y_spec)
    y_spec_ang = np.angle(y_spec)
    w = np.linspace(-np.pi/dt, np.pi/dt, 129); w = w[:-1]
    ii = np.where(y_spec_mag == np.max(y_spec_mag))
    est_w0, est_delta = w[ii][-1], y_spec_ang[ii][-1]
    print('Actual $\omega_0$ and estimated $\omega_0$ are {}, {}'.format(w0, est_w0))
    print('Actual $\delta$ and estimated $\delta$ are {}, {}'.format(delta, est_delta))

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(w, np.abs(y_spec), 'b', lw=2)
    # plt.xlim([-5,5])
    # plt.ylabel(r'$|Y|$', size=12)
    # plt.title(r'Spectrum of $cos(\omega_0 t + \delta)$')
    # plt.grid(True)
    # plt.subplot(2,1,2)
    # plt.plot(w, np.angle(y_spec), 'ro')
    # plt.xlim([-5,5])
    # plt.xlabel(r'$\omega$', size=12)
    # plt.ylabel(r'$\angle Y$', size=12)
    # plt.grid(True)

def q4():
    t, y, w0, delta = arbit_cos(noise=True)
    dt = t[1] - t[0]
    y[0] = 0
    y = fftshift(y)
    y_spec = fftshift(fft(y))/128
    y_spec_mag = np.abs(y_spec)
    y_spec_ang = np.angle(y_spec)
    w = np.linspace(-np.pi/dt, np.pi/dt, 129); w = w[:-1]
    ii = np.where(y_spec_mag == np.max(y_spec_mag))
    est_w0, est_delta = np.abs(w[ii][-1]), y_spec_ang[ii][-1]
    if est_delta < 0:
        est_delta += 2*np.pi
    print('Actual $\omega_0$ and estimated $\omega_0$ are {}, {}'.format(w0, est_w0))
    print('Actual $\delta$ and estimated $\delta$ are {}, {}'.format(delta, est_delta))

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(w, np.abs(y_spec), 'b', lw=2)
    # plt.xlim([-5,5])
    # plt.ylabel(r'$|Y|$', size=12)
    # plt.title(r'Spectrum of $cos(\omega_0 t + \delta)+$noise')
    # plt.grid(True)
    # plt.subplot(2,1,2)
    # plt.plot(w, np.angle(y_spec), 'ro')
    # plt.xlim([-5,5])
    # plt.xlabel(r'$\omega$', size=12)
    # plt.ylabel(r'$\angle Y$', size=12)
    # plt.grid(True)

def q5():
    t = np.linspace(-np.pi, np.pi, 1025); t = t[:-1]
    dt = t[1] - t[0]
    y = np.cos(16*(1.5 + t/(2*np.pi))*t)
    y[0] = 0
    # y = fftshift(y)
    y_spec = fftshift(fft(y))/1024
    w = np.linspace(-np.pi/dt, np.pi/dt, 1025); w = w[:-1]

    plt.figure()
    plt.plot(t, y, 'b')
    plt.xlabel(r'time', size=12)
    plt.ylabel(r'$y(t)$', size=12)
    plt.title(r'Chirp signal')
    plt.grid(True)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(w, np.abs(y_spec), 'b', lw=2)
    plt.xlim([-100,100])
    plt.ylabel(r'$|Y|$', size=12)
    plt.title(r'Spectrum of $cos(16(1.5 + t/2\pi)t)$')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(w, np.angle(y_spec), 'ro')
    plt.xlim([-100,100])
    plt.xlabel(r'$\omega$', size=12)
    plt.ylabel(r'$\angle Y$', size=12)
    plt.grid(True)

def q6():
    t = np.linspace(-np.pi, np.pi, 1025); t = t[:-1]
    dt = t[1] - t[0]
    y = np.cos(16*(1.5 + t/(2*np.pi))*t)
    y[0] = 0
    # y = fftshift(y)
    piece_length = 64
    freqvstime = np.zeros((piece_length, 1024-piece_length), dtype=complex)
    for i in range(1024-piece_length):
        piece_spec = fftshift(fft(y[i:i+piece_length]))/piece_length
        freqvstime[:,i] = piece_spec
    w = np.linspace(-np.pi/dt, np.pi/dt, piece_length+1); w = w[:-1]; w = w[28:-28]
    T, W = np.meshgrid(t[piece_length//2:-piece_length//2], w)

    # print(T.shape, W.shape, freqvstime.shape)

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    s = ax.plot_surface(T, W, np.abs(freqvstime)[28:-28,:], rstride=1, cstride=1, cmap=cm.jet)
    ax.set_xlabel(r'time t $\longrightarrow$')
    ax.set_ylabel(r'$\omega$ $\longrightarrow$')
    ax.set_title(r'Surface plot of frequency vs time')
    plt.colorbar(s, ax = ax, shrink=0.7)

    plt.figure()
    plt.contourf(T, W, np.abs(freqvstime)[28:-28,:], cmap=cm.jet)
    plt.xlabel(r'time', size=12)
    plt.ylabel(r'$\omega$', size=12)
    plt.title(r'Magnitude of spectrum vs time')
    plt.colorbar()

# ex1()
# ex2()
# ex3()
# ex4()
# ex5()
# ex6()
# ex7()
# q2()
# q3()
# q4()
# q5()
q6()

plt.show()