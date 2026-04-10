# ============================================================================
# Family 7 (Stochastic Oscillators) — Canonical 2-species Brusselator oscillator
# stochosc01_brusselator.py
# Two-species Brusselator reaction model exhibiting limit-cycle oscillations in the regime B > 1 + A^2.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "stochosc01_brusselator"

SPECIES_NAMES = [
    "X",  # autocatalytic species
    "Y"   # coupled species
]

PARAMS = {
    # Canonical Brusselator parameters (dimensionless concentrations)
    # Limit-cycle regime: B > 1 + A^2  (Hopf bifurcation)
    "A": 1.0,
    "B": 3.0
}

# Initial conditions are offset from the fixed point (X*=A, Y*=B/A) to make the limit cycle visible.
Y0 = [
    0.20,  # X
    5.00   # Y
]

TSPAN = (0.0, 60.0)

def dYdt(t, Y):
    p = PARAMS
    X, Yc = Y
    A = p["A"]
    B = p["B"]

    # Canonical Brusselator
    dX = A - (B + 1.0) * X + (X * X) * Yc
    dYc = B * X - (X * X) * Yc

    return np.array([dX, dYc], dtype=float)

if __name__ == "__main__":
    t_span = TSPAN
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau")

    # Micro-audit: per-species q80/q99 labels
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        y = sol.y[i]
        q80 = np.quantile(y, 0.80)
        q99 = np.quantile(y, 0.99)
        label = int((q80 < 200) and (q99 < 200))
        labels.append(label)
        print(f"{name}: q80={q80:.3f}, q99={q99:.3f} => label={label}")

    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass model")

    # Preview plot of species trajectories
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])

    plt.xlabel("t")
    plt.ylabel("abundance / concentration")
    plt.title(MODEL_NAME)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_preview.png", dpi=220)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
