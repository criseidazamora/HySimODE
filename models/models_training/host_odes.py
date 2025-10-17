import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===== PARAMETERS ======================================================
params = {
    'phie': 100, 'vT': 728, 'vE': 5800, 'kT': 1000, 'kE': 1000,
    'wT': 4.14, 'wE': 4.14, 'wH': 948.93, 'wR': 930, 'wr': 3170,
    'bT': 1, 'bE': 1, 'bH': 1, 'bR': 1,
    'oX': 4.38, 'oR': 426.87, 'uX': 1, 'dymX': 0.1,
    'nX': 300, 'nR': 7459, 'brho': 1, 'urho': 1,
    'kH': 152219, 'hH': 4, 'gammamax': 1260, 'gammakappa': 3e8,
    'M0': 1e8, 'abx': 0, 'wG': 0, 'bG': 1, 'dymG': 0.1, 'dypG': 0, 'nG': 300,
    'kin': 1e6, 'dil': 0.01
}

# ===== INITIAL CONDITIONS ============================================
Y0 = [
    128.800216874785, 399857681.007136,                         # iS, ee
    18.9335835404491, 14.3451993941180, 0, 509.900253355542,    # mT, cT, zT, pT
    18.9335835404491, 14.3451993941181, 0, 509.900253355544,    # mE, cE, zE, pE
    3679.21387779309, 2787.58939520311, 0, 99084.8924308509,    # mH, cH, zH, pH
    2907.49568398625, 6561.50164444711, 0, 0.0169230409369701,  # mR, cR, zR, pR
    37575.7368810624, 2.62691251598020,                         # rr, R
    0, 0, 0, 0,                                                 # mG, cG, zG, pG
    1e8, 1000                                                   # xS, N
]

# ===== SYSTEM ============================================
def host_odes(t, Y):
    p = params
    (iS, ee, mT, cT, zT, pT, mE, cE, zE, pE,
     mH, cH, zH, pH, mR, cR, zR, pR, rr, R,
     mG, cG, zG, pG, xS, N) = Y

    # Differential equations with inlined rates
    diS = (pT * p['vT'] * xS) / (p['kT'] + xS) - (pE * p['vE'] * iS) / (p['kE'] + iS) - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * iS

    dee = (p['phie'] * pE * p['vE'] * iS) / (p['kE'] + iS) - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * ee - \
          p['nR'] * ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nR'] * cR - \
          p['nX'] * ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * (cT + cE + cH) - \
          p['nG'] * ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nG'] * cG

    dmT = ((p['wT'] * ee) / (p['oX'] + ee)) - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG) + p['dymX']) * mT + \
          ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cT - p['bT'] * R * mT + p['uX'] * cT

    dcT = -((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * cT + \
          p['bT'] * R * mT - p['uX'] * cT - ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cT - p['abx'] * cT

    dzT = p['abx'] * cT - ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * zT

    dpT = ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cT - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * pT

    dmE = ((p['wE'] * ee) / (p['oX'] + ee)) - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG) + p['dymX']) * mE + \
          ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cE - p['bE'] * R * mE + p['uX'] * cE

    dcE = -((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * cE + \
          p['bE'] * R * mE - p['uX'] * cE - ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cE - p['abx'] * cE

    dzE = p['abx'] * cE - ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * zE

    dpE = ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cE - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * pE

    dmH = ((p['wH'] * ee) / (p['oX'] + ee)) / (1 + (pH / p['kH'])**p['hH']) - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG) + p['dymX']) * mH + \
          ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cH - p['bH'] * R * mH + p['uX'] * cH

    dcH = -((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * cH + \
          p['bH'] * R * mH - p['uX'] * cH - ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cH - p['abx'] * cH

    dzH = p['abx'] * cH - ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * zH

    dpH = ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nX'] * cH - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * pH

    dmR = ((p['wR'] * ee) / (p['oR'] + ee)) - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG) + p['dymX']) * mR + \
          ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nR'] * cR - p['bR'] * R * mR + p['uX'] * cR

    dcR = -((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * cR + \
          p['bH'] * R * mR - p['uX'] * cR - ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nR'] * cR - p['abx'] * cR

    dzR = p['abx'] * cR - ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * zR

    dpR = ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nR'] * cR - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * pR - \
          p['brho'] * pR * rr + p['urho'] * R

    drr = (p['wr'] * ee) / (p['oR'] + ee) - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * rr - \
          p['brho'] * pR * rr + p['urho'] * R

    dR = p['brho'] * pR * rr - p['urho'] * R - \
         ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * R + \
         ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) * (cT / p['nX'] + cE / p['nX'] + cH / p['nX'] + cR / p['nR'] + cG / p['nG']) - \
         (p['bT'] * R * mT + p['bE'] * R * mE + p['bH'] * R * mH + p['bR'] * R * mR + p['bG'] * R * mG) + \
         p['uX'] * (cT + cE + cH + cR + cG)

    dmG = ((p['wG'] * ee) / (p['oX'] + ee)) - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG) + p['dymG']) * mG + \
          ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nG'] * cG - p['bG'] * R * mG + p['uX'] * cG

    dcG = -((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * cG + \
          p['bG'] * R * mG - p['uX'] * cG - ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nG'] * cG - p['abx'] * cG

    dzG = p['abx'] * cG - ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG)) * zG

    dpG = ((p['gammamax'] * ee) / (p['gammakappa'] + ee)) / p['nG'] * cG - \
          ((1 / p['M0']) * (p['gammamax'] * ee) / (p['gammakappa'] + ee) * (cT + cE + cH + cR + cG) + p['dypG']) * pG

    dxS = -((p['vT'] * xS * pT) / (p['kT'] + xS)) * N
    if xS <= 0 and dxS < 0:
      dxS = 0.0

    dN = N * ee * p['gammamax'] * (cT + cE + cH + cR + cG) / (p['M0'] * (ee + p['gammakappa']))


    return [
        diS, dee, dmT, dcT, dzT, dpT, dmE, dcE, dzE, dpE,
        dmH, dcH, dzH, dpH, dmR, dcR, dzR, dpR, drr, dR,
        dmG, dcG, dzG, dpG, dxS, dN
    ]

# ===== SIMULATION ========================================================
# Usar Radau con tolerancias ajustadas
sol = solve_ivp(
    host_odes,
    t_span=(0, 1400),
    y0=Y0,
    t_eval=np.linspace(0, 1400, 1000),
    method='Radau',
    rtol=1e-6,
    atol=1e-9
)


# ===== PLOT =======================================================
plt.plot(sol.t, sol.y[25], label='Population N')
plt.xlabel("Tiempo [min]")
plt.ylabel("N (Population)")
plt.title("Growth (Batch)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("host.png", dpi=300)

# ===== SAVING DATA ==========================================
# List of variables
var_names = [
    "iS", "ee", "mT", "cT", "zT", "pT", "mE", "cE", "zE", "pE",
    "mH", "cH", "zH", "pH", "mR", "cR", "zR", "pR", "rr", "R",
    "mG", "cG", "zG", "pG", "xS", "N"
]

# End values of every variable
final_values = sol.y[:, -1]

# Create a dictionary for final results
final_state = dict(zip(var_names, final_values))

# Variables with less 200 molecules
low_molecules = {k: v for k, v in final_state.items() if v < 200}

# Show results
print("\n--- End time molecules species ---")
for k, v in final_state.items():
    print(f"{k}: {v:.2f}")

print("\n--- Variables - less 200 molecules ---")
for k, v in low_molecules.items():
    print(f"{k}: {v:.2f}")
