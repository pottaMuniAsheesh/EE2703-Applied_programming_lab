import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import sympy as sym

plt.rcParams.update({"text.usetex":True})

fignum = 1

def highpass(R1, R3, C1, C2, G, Vi):
    s = sym.symbols('s')
    A = sym.Matrix([[0,0,1,-1/G],[(s*C2*R3)/(1 + s*C2*R3),-1,0,0],[0,1,-1,0],[1/R1+s*C1+s*C2,-s*C2, 0, -1/R1]])
    b = sym.Matrix([0,0,0,Vi*s*C1])
    V = A.inv()*b
    return (A, b, V)

def lowpass(R1,R2,C1,C2,G,Vi):
    s = sym.symbols('s')
    A = sym.Matrix([[0,0,1,-1/G],[1/(1 + s*C2*R2),-1,0,0],[0,1,-1,0],[1/R1+1/R2+s*C1,-1/R2, 0, -s*C1]])
    b = sym.Matrix([0,0,0,Vi/R1])
    V = A.inv()*b
    return (A, b, V)

def freq_resp_lpf():
    s = sym.symbols('s')
    A, b, V = lowpass(10000, 10000, 1e-11, 1e-11, 1.586, 1)
    Vo = V[3]
    # print(Vo)
    num , den = Vo.as_numer_denom()
    num, den = np.array(sym.Poly(num, s).all_coeffs(), dtype=float), np.array(sym.Poly(den, s).all_coeffs(), dtype=float)
    # num, den = [1.586], [1e-10, 1.414e-5, 1]
    H = sp.lti(num, den)

    global fignum

    plt.figure(fignum)
    fignum += 1
    w = np.logspace(2,10,801)
    w, mag, phi = H.bode(w)
    plt.semilogx(w, mag)
    plt.grid(True)
    plt.title(r'Magnitude response of lowpass filter')
    plt.xlabel(r'$\omega$ (in rads/s)', size=12)
    plt.ylabel(r'$|H_{LPF}(j\omega)|$', size=12)

    plt.figure(fignum)
    fignum += 1
    plt.semilogx(w, phi)
    plt.grid(True)
    plt.title(r'Phase response of lowpass filter')
    plt.xlabel(r'$\omega$ (in rads/s)', size=12)
    plt.ylabel(r'$\angle H_{LPF}(j\omega)$', size=12)

def q1():
    s = sym.symbols('s')
    A, b, V = lowpass(10000, 10000, 1e-11, 1e-11, 1.586, 1/s)
    Vo = V[3]
    # print(Vo)
    num , den = Vo.as_numer_denom()
    num, den = np.array(sym.Poly(num, s).all_coeffs(), dtype=float), np.array(sym.Poly(den, s).all_coeffs(), dtype=float)
    # num, den = [1.586], [1e-10, 1.414e-5, 1]
    H = sp.lti(num, den)

    t = np.arange(0, 1e-5, 1e-8)
    t, y = sp.impulse(H, None, t)
    # w = np.logspace(0, 8, 801)
    # w, mag, phi = H.bode(w)
    global fignum
    plt.figure(fignum)
    fignum += 1
    plt.plot(t, y)
    plt.title(r'Step response ($s(t)$) of lowpass filter')
    plt.xlabel(r"time $t$ (in seconds)", size=12)
    plt.ylabel(r'$s(t)$', size=12)
    plt.grid(True)

    # plt.figure(fignum)
    # fignum += 1
    # plt.semilogx(w, mag)
    # plt.title('Magnitude response of lowpass filter')
    # plt.grid(True)

    # plt.figure(fignum)
    # fignum += 1
    # plt.semilogx(w, phi)
    # plt.title('Phase response of lowpass filter')
    # plt.grid(True)

def q2():
    s = sym.symbols('s')
    A, b, V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
    Vo = V[3]
    # print(Vo)
    num , den = Vo.as_numer_denom()
    num, den = np.array(sym.Poly(num, s).all_coeffs(), dtype=float), np.array(sym.Poly(den, s).all_coeffs(), dtype=float)
    H = sp.lti(num, den)

    t = np.arange(0, 10e-3, 1e-7)
    vi = np.sin(2*np.pi*1e3*t) + np.cos(2*np.pi*1e6*t)
    t, vo = sp.lsim(H, vi, t)[:2]
    global fignum
    plt.figure(fignum)
    fignum += 1
    plt.plot(t[:100], vo[:100])
    plt.title(r'Output voltage $v_o(t)$')
    plt.xlabel(r'time $t$ (in seconds)', size=12)
    plt.ylabel(r'$v_o(t)$', size=12)
    plt.grid(True)

    plt.figure(fignum)
    fignum += 1
    plt.plot(t, vo)
    plt.title(r'Output voltage $v_o(t)$')
    plt.xlabel(r'time $t$ (in seconds)', size=12)
    plt.ylabel(r'$v_o(t)$', size=12)
    plt.grid(True)

    plt.figure(fignum)
    fignum += 1
    plt.plot(t[:100], vi[:100])
    plt.title(r'Input signal $v_i(t) = (sin(2000 \pi t)+cos(2 \times 10^6 \pi t))u(t)$')
    plt.xlabel(r'time $t$ (in seconds)', size=12)
    plt.ylabel(r'$v_i(t)$', size=12)
    plt.grid(True)

    plt.figure(fignum)
    fignum += 1
    plt.plot(t, vi)
    plt.title(r'Input signal $v_i(t) = (sin(2000 \pi t)+cos(2 \times 10^6 \pi t))u(t)$')
    plt.xlabel(r'time $t$ (in seconds)', size=12)
    plt.ylabel(r'$v_i(t)$', size=12)
    plt.grid(True)

def q3():
    s = sym.symbols('s')
    A, b, V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
    Vo = V[3]
    # print(Vo)
    num , den = Vo.as_numer_denom()
    num, den = np.array(sym.Poly(num, s).all_coeffs(), dtype=float), np.array(sym.Poly(den, s).all_coeffs(), dtype=float)
    H = sp.lti(num, den)

    global fignum

    plt.figure(fignum)
    fignum += 1
    w = np.logspace(0,8,801)
    w, mag, phi = H.bode(w)
    plt.semilogx(w, mag)
    plt.grid(True)
    plt.title(r'Magnitude response of highpass filter')
    plt.xlabel(r'$\omega$ (in rads/s)', size=12)
    plt.ylabel(r'$|H_{HPF}(j\omega)|$ (in dB)', size=12)

    plt.figure(fignum)
    fignum += 1
    plt.semilogx(w, phi)
    plt.grid(True)
    plt.title(r'Phase response of highpass filter')
    plt.xlabel(r'$\omega$ (in rads/s)', size=12)
    plt.ylabel(r'$\angle H_{HPF}(j\omega)$ (in degrees)', size=12)

def q4():
    s = sym.symbols('s')
    a, w0 = 1e1, 1e3
    Vi = (s+a)/((s+a)**2 + w0**2)
    A, b, V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, Vi)
    Vo = V[3]
    # print(Vo)
    Vi_TF = sp.lti([1, a], [1, 2*a, w0**2 + a**2])
    num , den = Vo.as_numer_denom()
    num, den = np.array(sym.Poly(num, s).all_coeffs(), dtype=float), np.array(sym.Poly(den, s).all_coeffs(), dtype=float)
    H = sp.lti(num, den)

    t = np.arange(0, 1, 1e-4)
    vi = np.exp(-a*t)*np.cos(w0*t)
    # t, vi = sp.impulse(Vi_TF, None, t)
    t, h = sp.impulse(H, None, t)
    global fignum
    plt.figure(fignum)
    fignum += 1
    plt.plot(t,vi)
    plt.title(r'Input signal $v_i(t) = exp(-10 t)cos(10^3 t)u(t)$')
    plt.xlabel(r'time $t$ (in seconds)', size=12)
    plt.ylabel(r'$v_i(t)$', size=12)
    plt.grid(True)

    plt.figure(fignum)
    fignum += 1
    plt.plot(t[:100],vi[:100])
    plt.title(r'Input signal $v_i(t) = exp(-10 t)cos(10^3 t)u(t)$')
    plt.xlabel(r'time $t$ (in seconds)', size=12)
    plt.ylabel(r'$v_i(t)$', size=12)
    plt.grid(True)

    plt.figure(fignum)
    fignum += 1
    plt.plot(t, h)
    plt.title(r'Output signal')
    plt.xlabel(r'time $t$ (in seconds)', size=12)
    plt.ylabel(r'$v_o(t)$', size=12)
    plt.grid(True)

    plt.figure(fignum)
    fignum += 1
    plt.plot(t[:100], h[:100])
    plt.title(r'Output signal')
    plt.xlabel(r'time $t$ (in seconds)', size=12)
    plt.ylabel(r'$v_o(t)$', size=12)
    plt.grid(True)

def q5():
    s = sym.symbols('s')
    A, b, V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1/s)
    Vo = V[3]
    # print(Vi)
    # Vi_TF = sp.lti([1, 0.01],[1, 2e-2, 1e7])
    num , den = Vo.as_numer_denom()
    num, den = np.array(sym.Poly(num, s).all_coeffs(), dtype=float), np.array(sym.Poly(den, s).all_coeffs(), dtype=float)
    H = sp.lti(num, den)

    t = np.arange(0, 1e-3, 1e-7)
    t, h = sp.impulse(H, None, t)
    global fignum
    plt.figure(fignum)
    fignum += 1
    plt.plot(t, h)
    plt.title(r'Step response of Highpass filter')
    plt.xlabel(r"time $t$ (in seconds)", size=12)
    plt.ylabel(r'$s(t)$', size=12)
    plt.grid(True)

# freq_resp_lpf()
# q1()
# q2()
# q3()
# q4()
q5()

plt.show()