# ============================================================================
# Family 4 (Deterministic Oscillators) — Goodwin oscillator with distributed delay
# osc03_goodwin_oscillator.py
# Goodwin-type negative-feedback oscillator with high-cooperativity transcriptional
# repression, linear degradation, and a five-step delay chain generating a
# distributed feedback delay.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "osc03_goodwin_oscillator"

SPECIES_NAMES = [
    "mRNA",  # S0
    "P0",    # S1
    "P1",    # S2
    "P2",    # S3
    "P3",    # S4
    "P4",    # S5
    "Rep",   # S6
]

PARAMS = {
    # Transcriptional repression
    "alpha_max": 180.0,   # molecules/min
    "alpha_leak": 0.25,   # molecules/min  (strong fold-change)
    "K_rep": 900.0,       # molecules
    "n_rep": 12.0,        # >= 10

    # Linear degradation
    "delta_m": 0.030,     # /min
    "delta_p": 0.0016,    # /min
    "delta_rep": 0.0016,  # /min

    # Translation 
    "k_tl": 0.55,         # molecules/min per mRNA

    # Five-step distributed delay chain
    "k01": 0.0038,        # /min
    "k12": 0.0038,        # /min
    "k23": 0.0038,        # /min
    "k34": 0.0038,        # /min
    "k45": 0.0038,        # /min
}

# Initial conditions in an oscillatory regime
Y0 = [
    90.0,     # mRNA
    6000.0,   # P0
    3500.0,   # P1
    2200.0,   # P2
    1600.0,   # P3
    1100.0,   # P4
    900.0,    # Rep
]

TSPAN = (0.0, 6000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    m, P0, P1, P2, P3, P4, Rep = Y

    # Hill repression (high cooperativity)
    rep_term = 1.0 / (1.0 + (Rep / p["K_rep"]) ** p["n_rep"])
    v_tx = p["alpha_leak"] + p["alpha_max"] * rep_term
    v_mdeg = p["delta_m"] * m

    # Translation and delay chain
    v_tl = p["k_tl"] * m
    v01 = p["k01"] * P0
    v12 = p["k12"] * P1
    v23 = p["k23"] * P2
    v34 = p["k34"] * P3
    v45 = p["k45"] * P4

    # Linear degradation
    v_P0_deg = p["delta_p"] * P0
    v_P1_deg = p["delta_p"] * P1
    v_P2_deg = p["delta_p"] * P2
    v_P3_deg = p["delta_p"] * P3
    v_P4_deg = p["delta_p"] * P4
    v_Rep_deg = p["delta_rep"] * Rep

    dm = v_tx - v_mdeg
    dP0 = v_tl - v01 - v_P0_deg
    dP1 = v01 - v12 - v_P1_deg
    dP2 = v12 - v23 - v_P2_deg
    dP3 = v23 - v34 - v_P3_deg
    dP4 = v34 - v45 - v_P4_deg
    dRep = v45 - v_Rep_deg

    return np.array([dm, dP0, dP1, dP2, dP3, dP4, dRep], dtype=float)

if __name__ == "__main__":
    t_span = TSPAN
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    # Micro-audit
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        y = sol.y[i]
        q80 = np.quantile(y, 0.80)
        q99 = np.quantile(y, 0.99)
        label = int((q80 < 200) and (q99 < 200))
        labels.append(label)
        print(f"{name}: q80={q80:.2f}, q99={q99:.2f}, label={label}")

    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass model (may not be useful for training)")

    # Preview plot of species trajectories
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("osc03_goodwin_oscillator (5-step delay chain)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("osc03_goodwin_oscillator_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
