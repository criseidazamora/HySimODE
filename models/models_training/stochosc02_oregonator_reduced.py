import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ======================================================================
# Family 7 — Noise-Prone Oscillatory Models
# Model 35 (retuned): stochosc02_oregonator_reduced
# Reduced 3-species Oregonator consistent with the classic FKN-derived scheme:
#   A + Y -> X + P
#   X + Y -> 2P
#   A + X -> 2X + 2Z
#   2X -> A + P
#   B + Z -> (f/2) Y
# leading to:
#   dX/dt = k1 A Y - k2 X Y + k3 A X - 2 k4 X^2
#   dY/dt = -k1 A Y - k2 X Y + (f/2) k5 B Z
#   dZ/dt = 2 k3 A X - k5 B Z
#
# Retuned to produce sustained oscillations and mixed ML labels (>=1 label0).
# ======================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ======================================================================
# Family 7 — Noise-Prone Oscillatory Models
# Model 35: stochosc02_oregonator_reduced
# Reduced 3-species Oregonator consistent with the classic FKN-derived scheme:
#   A + Y -> X + P
#   X + Y -> 2P
#   A + X -> 2X + 2Z
#   2X -> A + P
#   B + Z -> (f/2) Y
# leading to:
#   dX/dt = k1 A Y - k2 X Y + k3 A X - 2 k4 X^2
#   dY/dt = -k1 A Y - k2 X Y + (f/2) k5 B Z
#   dZ/dt = 2 k3 A X - k5 B Z
#
# Implemented in dimensionless concentrations (x,y,z) but stored as
# molecule-like abundances via scaling u=sX*x, v=sY*y, w=sZ*z.
# ======================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ======================================================================
# Family 7 — Noise-Prone Oscillatory Models
# Model 35 (retuned): stochosc02_oregonator_reduced
# ======================================================================

MODEL_NAME = "stochosc02_oregonator_reduced"

SPECIES_NAMES = [
    "X_activator",
    "Y_inhibitor",
    "Z_oxidized_cat"
]

PARAMS = {
    "A": 1.0,
    "B": 1.0,

    "k1": 2.6,
    "k2": 300.0,
    "k3": 2.5,
    "k4": 0.08,
    "k5": 1.5,
    "f": 1.40,

    "sX": 120.0,
    "sY": 120.0,
    "sZ": 2500.0
}

Y0 = [
    0.30 * PARAMS["sX"],
    0.40 * PARAMS["sY"],
    0.25 * PARAMS["sZ"]
]

TSPAN = (0.0, 300.0)


def dYdt(t, Y):
    p = PARAMS
    u, v, w = Y

    x = u / p["sX"]
    y = v / p["sY"]
    z = w / p["sZ"]

    A = p["A"];  B = p["B"]
    k1=p["k1"]; k2=p["k2"]; k3=p["k3"]
    k4=p["k4"]; k5=p["k5"]; ff=p["f"]

    dx = (k1*A*y) - (k2*x*y) + (k3*A*x) - (2*k4*x*x)
    dy = -(k1*A*y) - (k2*x*y) + 0.5*ff*k5*B*z
    dz = (2*k3*A*x) - (k5*B*z)

    du = p["sX"] * dx
    dv = p["sY"] * dy
    dw = p["sZ"] * dz

    return np.array([du, dv, dw], dtype=float)


if __name__ == "__main__":
    t_span = TSPAN
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau")

    # ============================
    # Micro-audit
    # ============================
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

    # ============================
    # Plot 1: X and Y (low-copy species)
    # ============================
    plt.figure(figsize=(8,5))
    plt.plot(sol.t, sol.y[0], label="X_activator")
    plt.plot(sol.t, sol.y[1], label="Y_inhibitor")
    plt.xlabel("t")
    plt.ylabel("abundance / concentration")
    plt.title(MODEL_NAME + " — X & Y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_XY_preview.png", dpi=220)

    # ============================
    # Plot 2: Z only (high-copy species)
    # ============================
    plt.figure(figsize=(8,5))
    plt.plot(sol.t, sol.y[2], label="Z_oxidized_cat", color="green")
    plt.xlabel("t")
    plt.ylabel("abundance / concentration")
    plt.title(MODEL_NAME + " — Z only")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_Z_preview.png", dpi=220)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
