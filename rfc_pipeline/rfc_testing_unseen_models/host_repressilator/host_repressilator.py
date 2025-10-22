# host_repressilator.py
# Pipeline:
#   1) WarmUp1 (host, 22 vars)
#   2) WarmUp2 (host+GFP, 25 vars)
#   3) Host+Repressilator (31 vars)
# - Solver: SciPy solve_ivp(method="BDF")
# - EPS break simmetry to (g1, g2, g3), (mg1, mg2, mg3) and (rmg1, rmg2, rmg3)

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# user options
# =========================
RTOL = 1e-6
ATOL = 1e-8
PLOT = True         # True for plot g1,g2,g3
EPS  = 0.10          

# =========================
# Parameters (warmUp1/2)
# =========================
thetar = 426.8693338968694
k_cm   = 0.005990373118888
nr     = 7549.0
gmax   = 1260.0
cl     = 0.0
nume0  = 4.139172187824451
s0     = 1.0e4
vm     = 5800.0
Km     = 1.0e3
numr0  = 929.9678874564831
nx     = 300.0
kq     = 1.522190403737490e5
Kp     = 180.1378030928276
vt     = 726.0
nump0  = 0.0
numq0  = 948.9349882947897
Kt     = 1.0e3
nq     = 4.0
aatot  = 1.0e8
ns     = 0.5
thetax = 4.379733394834643

# Rates
b = 0.0
dm = 0.1
kb = 1.0
ku = 1.0
f  = cl * k_cm  # f=0 con cl=0
dmg = np.log(2)/2.0
dg  = np.log(2)/4.0

# =========================================================
# WarmUp1 ODEs (22 vars)  
# y = [rmr, em, rmp, rmq, rmt, et, rmm, zmm, zmr, zmp, zmq, zmt,
#      mt,  mm,  q,   p,   si,  mq,  mp,  mr,  r,   a]
# =========================================================
def warmup1_odes(t, y):
    (rmr, em, rmp, rmq, rmt, et, rmm, zmm, zmr, zmp, zmq, zmt,
     mt, mm, q, p, si, mq, mp, mr, r, a) = np.maximum(y, 0.0)

    Kg    = gmax / Kp
    gamma = gmax * a / (Kg + a)
    ttrate= (rmq + rmr + rmp + rmt + rmm) * gamma
    lam   = ttrate / aatot
    nucat = em * vm * si / (Km + si)

    dyrmr = +kb*r*mr + b*zmr - ku*rmr - gamma/nr*rmr - f*rmr - lam*rmr
    dyem  = +gamma/nx*rmm - lam*em
    dyrmp = +kb*r*mp + b*zmp - ku*rmp - gamma/nx*rmp - f*rmp - lam*rmp
    dyrmq = +kb*r*mq + b*zmq - ku*rmq - gamma/nx*rmq - f*rmq - lam*rmq
    dyrmt = +kb*r*mt + b*zmt - ku*rmt - gamma/nx*rmt - f*rmt - lam*rmt
    dyet  = +gamma/nx*rmt - lam*et
    dyrmm = +kb*r*mm + b*zmm - ku*rmm - gamma/nx*rmm - f*rmm - lam*rmm
    dyzmm = +f*rmm - b*zmm - lam*zmm
    dyzmr = +f*rmr - b*zmr - lam*zmr
    dyzmp = +f*rmp - b*zmp - lam*zmp
    dyzmq = +f*rmq - b*zmq - lam*zmq
    dyzmt = +f*rmt - b*zmt - lam*zmt
    dymt  = +(nume0*a/(thetax+a)) + ku*rmt + gamma/nx*rmt - kb*r*mt - dm*mt - lam*mt
    dymm  = +(nume0*a/(thetax+a)) + ku*rmm + gamma/nx*rmm - kb*r*mm - dm*mm - lam*mm
    dyq   = +gamma/nx*rmq - lam*q
    dyp   = +gamma/nx*rmp - lam*p
    dysi  = +(et*vt*s0/(Kt+s0)) - nucat - lam*si
    dymq  = +(numq0*a/(thetax+a)/(1+(q/kq)**nq)) + ku*rmq + gamma/nx*rmq - kb*r*mq - dm*mq - lam*mq
    dymp  = +(nump0*a/(thetax+a)) + ku*rmp + gamma/nx*rmp - kb*r*mp - dm*mp - lam*mp
    dymr  = +(numr0*a/(thetar+a)) + ku*rmr + gamma/nr*rmr - kb*r*mr - dm*mr - lam*mr
    dyr   = +ku*(rmr+rmt+rmm+rmp+rmq) \
            + gamma/nr*rmr + gamma/nr*rmr \
            + gamma/nx*(rmt+rmm+rmp+rmq) \
            - kb*r*(mr+mt+mm+mp+mq) - lam*r
    dya   = +ns*nucat - ttrate - lam*a

    return np.array([
        dyrmr, dyem, dyrmp, dyrmq, dyrmt,
        dyet,  dyrmm, dyzmm, dyzmr, dyzmp,
        dyzmq, dyzmt, dymt,  dymm,  dyq,
        dyp,   dysi,  dymq,  dymp,  dymr,
        dyr,   dya
    ], dtype=float)

# =========================================================
# WarmUp2 ODEs (25 vars) 
# y = [rmr, em, rmp, rmq, rmt, rmg, et, rmm, zmm, zmr, zmp, zmq, zmt,
#      mt,  mg,  g,   mm,  q,   p,   si,  mq,  mp,  mr,  r,  a]
# =========================================================
def warmup2_gfp_odes(t, y):
    (rmr, em, rmp, rmq, rmt, rmg, et, rmm, zmm, zmr, zmp, zmq, zmt,
     mt, mg, g, mm, q, p, si, mq, mp, mr, r, a) = np.maximum(y, 0.0)

    Kg    = gmax / Kp
    gamma = gmax * a / (Kg + a)
    ttrate= (rmq + rmr + rmp + rmt + rmm + rmg) * gamma
    lam   = ttrate / aatot
    nucat = em * vm * si / (Km + si)

    dyrmr = +kb*r*mr + b*zmr - ku*rmr - gamma/nr*rmr - f*rmr - lam*rmr
    dyem  = +gamma/nx*rmm - lam*em
    dyrmp = +kb*r*mp + b*zmp - ku*rmp - gamma/nx*rmp - f*rmp - lam*rmp
    dyrmq = +kb*r*mq + b*zmq - ku*rmq - gamma/nx*rmq - f*rmq - lam*rmq
    dyrmt = +kb*r*mt + b*zmt - ku*rmt - gamma/nx*rmt - f*rmt - lam*rmt
    dyrmg = +kb*r*mg - ku*rmg - gamma/nx*rmg - lam*rmg
    dyet  = +gamma/nx*rmt - lam*et
    dyrmm = +kb*r*mm + b*zmm - ku*rmm - gamma/nx*rmm - f*rmm - lam*rmm
    dyzmm = +f*rmm - b*zmm - lam*zmm
    dyzmr = +f*rmr - b*zmr - lam*zmr
    dyzmp = +f*rmp - b*zmp - lam*zmp
    dyzmq = +f*rmq - b*zmq - lam*zmq
    dyzmt = +f*rmt - b*zmt - lam*zmt
    dymt  = +(nume0*a/(thetax+a)) + ku*rmt + gamma/nx*rmt - kb*r*mt - dm*mt - lam*mt
    dymg  = +(25.0*a/(thetax+a)) + ku*rmg + gamma/nx*rmg - kb*r*mg - dmg*mg - lam*mg
    dyg   = +gamma/nx*rmg - dg*g - lam*g
    dymm  = +(nume0*a/(thetax+a)) + ku*rmm + gamma/nx*rmm - kb*r*mm - dm*mm - lam*mm
    dyq   = +gamma/nx*rmq - lam*q
    dyp   = +gamma/nx*rmp - lam*p
    dysi  = +(et*vt*s0/(Kt+s0)) - nucat - lam*si
    dymq  = +(numq0*a/(thetax+a)/(1+(q/kq)**nq)) + ku*rmq + gamma/nx*rmq - kb*r*mq - dm*mq - lam*mq
    dymp  = +(nump0*a/(thetax+a)) + ku*rmp + gamma/nx*rmp - kb*r*mp - dm*mp - lam*mp
    dymr  = +(numr0*a/(thetar+a)) + ku*rmr + gamma/nr*rmr - kb*r*mr - dm*mr - lam*mr
    dyr   = +ku*(rmr+rmt+rmm+rmp+rmq+rmg) \
            + gamma/nr*rmr + gamma/nr*rmr \
            + gamma/nx*(rmt+rmm+rmp+rmq+rmg) \
            - kb*r*(mr+mt+mm+mp+mq+mg) - lam*r
    dya   = +ns*nucat - ttrate - lam*a

    return np.array([
        dyrmr, dyem,  dyrmp, dyrmq,  dyrmt,
        dyrmg, dyet,  dyrmm, dyzmm,  dyzmr,
        dyzmp, dyzmq, dyzmt, dymt,   dymg,
        dyg,   dymm,  dyq,   dyp,    dysi,
        dymq,  dymp,  dymr,  dyr,    dya
    ], dtype=float)

# =========================================================
# Host + Repressilator ODEs (31 vars) 
# y = [rmr, em, rmp, rmq, rmt, mg3, mg2, mg1, et, rmm,
#      zmm, zmr, rmg2, zmp, zmq, zmt, rmg3, g3, g2, g1,
#      rmg1, mt, mm, q, p, si, mq, mp, mr, r, a]
# =========================================================
def repressilator_odes(t, y):
    (rmr, em, rmp, rmq, rmt,
     mg3, mg2, mg1, et, rmm,
     zmm, zmr, rmg2, zmp, zmq,
     zmt, rmg3, g3, g2, g1,
     rmg1, mt, mm, q, p, si, mq, mp, mr, r, a) = np.maximum(y, 0.0)

    Kg    = gmax / Kp
    gamma = gmax * a / (Kg + a)
    ttrate= (rmq+rmr+rmp+rmt+rmm+rmg1+rmg2+rmg3) * gamma
    lam   = ttrate / aatot
    nucat = em * vm * si / (Km + si)

    #repressilator parameters
    hcoeff = 2.0
    th     = 100.0
    kf     = 1.0
    numg0  = 25.0

    ff1 = kf/(1.0 + (g3/th)**hcoeff)  # → mg1
    ff2 = kf/(1.0 + (g1/th)**hcoeff)  # → mg2
    ff3 = kf/(1.0 + (g2/th)**hcoeff)  # → mg3

    # Derivatives
    dyrmr = +kb*r*mr + b*zmr - ku*rmr - gamma/nr*rmr - f*rmr - lam*rmr
    dyem  = +gamma/nx*rmm - lam*em
    dyrmp = +kb*r*mp + b*zmp - ku*rmp - gamma/nx*rmp - f*rmp - lam*rmp
    dyrmq = +kb*r*mq + b*zmq - ku*rmq - gamma/nx*rmq - f*rmq - lam*rmq
    dyrmt = +kb*r*mt + b*zmt - ku*rmt - gamma/nx*rmt - f*rmt - lam*rmt

    dymg3 = +(ff3*numg0*a/(thetax+a)) + ku*rmg3 + gamma/nx*rmg3 - kb*r*mg3 - dmg*mg3 - lam*mg3
    dymg2 = +(ff2*numg0*a/(thetax+a)) + ku*rmg2 + gamma/nx*rmg2 - kb*r*mg2 - dmg*mg2 - lam*mg2
    dymg1 = +(ff1*numg0*a/(thetax+a)) + ku*rmg1 + gamma/nx*rmg1 - kb*r*mg1 - dmg*mg1 - lam*mg1

    dyet  = +gamma/nx*rmt - lam*et
    dyrmm = +kb*r*mm + b*zmm - ku*rmm - gamma/nx*rmm - f*rmm - lam*rmm
    dyzmm = +f*rmm - b*zmm - lam*zmm
    dyzmr = +f*rmr - b*zmr - lam*zmr
    dyrmg2= +kb*r*mg2 - ku*rmg2 - gamma/nx*rmg2 - lam*rmg2
    dyzmp = +f*rmp - b*zmp - lam*zmp
    dyzmq = +f*rmq - b*zmq - lam*zmq
    dyzmt = +f*rmt - b*zmt - lam*zmt
    dyrmg3= +kb*r*mg3 - ku*rmg3 - gamma/nx*rmg3 - lam*rmg3
    dyg3  = +gamma/nx*rmg3 - dg*g3 - lam*g3
    dyg2  = +gamma/nx*rmg2 - dg*g2 - lam*g2
    dyg1  = +gamma/nx*rmg1 - dg*g1 - lam*g1
    dyrmg1= +kb*r*mg1 - ku*rmg1 - gamma/nx*rmg1 - lam*rmg1

    dymt  = +(nume0*a/(thetax+a)) + ku*rmt + gamma/nx*rmt - kb*r*mt - dm*mt - lam*mt
    dymm  = +(nume0*a/(thetax+a)) + ku*rmm + gamma/nx*rmm - kb*r*mm - dm*mm - lam*mm
    dyq   = +gamma/nx*rmq - lam*q
    dyp   = +gamma/nx*rmp - lam*p
    dysi  = +(et*vt*s0/(Kt+s0)) - nucat - lam*si
    dymq  = +(numq0*a/(thetax+a)/(1+(q/kq)**nq)) + ku*rmq + gamma/nx*rmq - kb*r*mq - dm*mq - lam*mq
    dymp  = +(nump0*a/(thetax+a)) + ku*rmp + gamma/nx*rmp - kb*r*mp - dm*mp - lam*mp
    dymr  = +(numr0*a/(thetar+a)) + ku*rmr + gamma/nr*rmr - kb*r*mr - dm*mr - lam*mr
    dyr   = +ku*(rmr+rmt+rmm+rmp+rmq+rmg1+rmg2+rmg3) \
            + gamma/nr*rmr + gamma/nr*rmr \
            + gamma/nx*(rmt+rmm+rmp+rmq+rmg1+rmg2+rmg3) \
            - kb*r*(mr+mt+mm+mp+mq+mg1+mg2+mg3) - lam*r
    dya   = +ns*nucat - ttrate - lam*a

    return np.array([
        dyrmr, dyem,  dyrmp,  dyrmq,  dyrmt,
        dymg3, dymg2, dymg1,  dyet,   dyrmm,
        dyzmm, dyzmr, dyrmg2, dyzmp,  dyzmq,
        dyzmt, dyrmg3, dyg3,  dyg2,   dyg1,
        dyrmg1, dymt, dymm,   dyq,    dyp,
        dysi,   dymq, dymp,   dymr,   dyr, dya
    ], dtype=float)

# =========================
# Helper (BDF)
# =========================
def integrate(fun, y0, tmax, n_eval=400):
    t_eval = np.linspace(0.0, tmax, n_eval)
    sol = solve_ivp(fun, (0.0, tmax), y0, method="BDF",
                    rtol=RTOL, atol=ATOL, t_eval=t_eval)
    return sol

# =========================
# Pipeline - principal
# =========================
def main():
    # --- WarmUp1 initial (22 vars)
    y0_w1 = np.array([
        0,0,0,0,0,   # rmr em rmp rmq rmt
        0,0,0,0,0,   # et rmm zmm zmr zmp
        0,0,         # zmq zmt
        0,0,0,0,0,   # mt mm q p si
        0,0,0,       # mq mp mr
        10.0, 1000.0 # r a
    ], dtype=float)

    sol1 = integrate(warmup1_odes, y0_w1, tmax=1e7)
    w1 = sol1.y[:, -1]

    # --- WarmUp2 initial (25 vars) from WarmUp1 
    (rmr, em, rmp, rmq, rmt, et, rmm, zmm, zmr, zmp, zmq, zmt,
     mt, mm, q, p, si, mq, mp, mr, r, a) = w1
    mg0, rmg0, g0 = mm, rmm, em   # sin ruido, solo matching del warmUp2
    y0_w2 = np.array([
        rmr, em, rmp, rmq, rmt, rmg0, et, rmm, zmm, zmr,
        zmp, zmq, zmt, mt, mg0, g0, mm, q, p, si,
        mq, mp, mr, r, a
    ], dtype=float)

    sol2 = integrate(warmup2_gfp_odes, y0_w2, tmax=1e7)
    w2 = sol2.y[:, -1]

    # --- y0 (31 vars) for Host+Repressilator
    (rmr, em, rmp, rmq, rmt, rmg, et, rmm, zmm, zmr, zmp, zmq, zmt,
     mt, mg, g, mm, q, p, si, mq, mp, mr, r, a) = w2

    mg1_0  = mg*(1+EPS); mg2_0 = mg; mg3_0 = mg*(1-EPS)
    rmg1_0 = rmg*(1+EPS); rmg2_0 = rmg; rmg3_0 = rmg*(1-EPS)
    g1_0   = g*(1+EPS);   g2_0   = g;  g3_0   = g*(1-EPS)

    y0 = np.array([
        rmr, em, rmp, rmq, rmt,
        mg3_0, mg2_0, mg1_0, et, rmm,
        zmm, zmr, rmg2_0, zmp, zmq,
        zmt, rmg3_0, g3_0, g2_0, g1_0,
        rmg1_0, mt, mm, q, p, si, mq, mp, mr, r, a
    ], dtype=float)

    # --- Integrate host+repressilator
    t_eval = np.arange(0.0, 5000.0+1.0, 1.0)
    sol3 = solve_ivp(repressilator_odes, (0.0, 5000.0), y0,
                     method="BDF", rtol=RTOL, atol=ATOL, t_eval=t_eval)

    if PLOT:
        g1 = sol3.y[19]; g2 = sol3.y[18]; g3 = sol3.y[17]
        plt.figure(figsize=(10,5))
        plt.plot(sol3.t, g1, label='g1')
        plt.plot(sol3.t, g2, label='g2')
        plt.plot(sol3.t, g3, label='g3')
        plt.xlabel('time'); plt.ylabel('concentration')
        plt.title('Host–Repressilator (31 vars, BDF)')
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig("host.png", dpi=300)
        plt.close()

    # Return y0 (31 vars) 
    return y0, sol3

if __name__ == "__main__":
    y0, sol = main()
    print("y0 (31 vars) ready for test.", y0[:6])
