import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

fignum = 1

def q1_q2(a=0.5):
    Num = np.array([1, a])
    Den = np.polymul([1, 2*a, 2.25+(a*a)], [1, 0, 2.25])
    H = sp.lti(Num, Den)
    t, x = sp.impulse(H, None, np.linspace(0, 50, 501))
    global fignum
    plt.figure(fignum)
    fignum += 1
    plt.plot(t, x)
    plt.title(r'Undamped forced oscillation with $f(t) = cos(1.5t) e^{-at} u_0(t)$')

def q3():
    H = sp.lti([1], [1, 0, 2.25])
    global fignum
    t = np.linspace(0, 50, 501)
    for w in np.arange(1.4, 1.6, 0.05):
        f = np.cos(w*t)*np.exp(-0.05*t)
        t, x = sp.lsim(H, f, t)[:2]
        plt.figure(fignum)
        fignum += 1
        plt.plot(t, x, label=r'$\omega$'+' = %.2f'%(w))
        plt.title(r'Variation with frequency')
        plt.legend()

def q4():
    X = sp.lti([1, 0, 2], [1, 0, 3, 0])
    Y = sp.lti([2], [1, 0, 3, 0])
    t = np.linspace(0, 20, 201)
    t, x = sp.impulse(X, None, t)
    t, y = sp.impulse(Y, None, t)
    global fignum
    plt.figure(fignum)
    fignum += 1
    plt.plot(t, x, 'r', label='x(t)')
    plt.plot(t, y, 'g', label='y(t)')
    plt.title('Q4: Solutions of the coupled spring problem')
    plt.legend()

def q5_q6():
    H = sp.lti([1], [1e-12, 1e-4, 1])
    w, mag, phi = H.bode()
    global fignum
    plt.figure(fignum,figsize=(8,6))
    fignum += 1
    plt.subplot(2, 1, 1)
    # plt.tight_layout()
    plt.title('Magnitude response of RLC circuit')
    plt.semilogx(w, mag)
    plt.grid(True)
    plt.ylabel('Magnitude (in dB)')
    plt.subplot(2, 1, 2)
    plt.tight_layout()
    plt.title('Phase response of RLC circuit')
    plt.semilogx(w, phi)
    plt.grid(True)
    plt.ylabel('Phase (in degrees)')
    plt.xlabel('Frequency (in rad/s)')

    t = np.arange(0, 1e-2, 1e-7)
    vi = np.cos(1e3*t) - np.cos(1e6*t)
    t, vo = sp.lsim(H, vi, t)[:2]
    plt.figure(fignum)
    fignum += 1
    plt.plot(t[:300], vo[:300], label=r'$v_o(t)$')
    plt.grid(True)
    plt.legend()
    plt.title(r'Response for t < 30$\mu$s')
    plt.figure(fignum)
    fignum += 1
    plt.plot(t, vo, label=r'$v_o(t)$')
    plt.grid(True)
    plt.legend()
    plt.title(r'Response for t < 10ms')

q1_q2(0.5)
q1_q2(0.05)
q3()
q4()
q5_q6()

plt.show()