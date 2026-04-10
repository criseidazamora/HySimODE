# ============================================================================
# Family 7 (Stochastic Oscillators) — FitzHugh–Nagumo-like biochemical oscillator
# stochosc04_fitzhugh_nagumo_like.py
# FitzHugh–Nagumo-like relaxation oscillator mapped to nonnegative biochemical 
# observables and coupled to nutrient-pool and product dynamics.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "stochosc04_fitzhugh_nagumo_like"
SPECIES_NAMES = ["A_activator", "I_inhibitor", "N_pool", "P_product"]

# Model structure:
#   The core dynamics follow a FitzHugh–Nagumo relaxation oscillator (u, v).
#   These internal variables are mapped to positive biochemical observables:
#       A_activator = A_scale * (u + u0)
#       I_inhibitor = I_scale * (v + v0)
#   Additional variables represent metabolic context:
#       N_pool: buffered nutrient reservoir
#       P_product: product generated from nutrient consumption

PARAMS = {
    # --- FHN core (dimensionless) ---
    # These values are in a known limit-cycle regime for the FHN equations.
    "a": 0.70,
    "b": 0.80,
    "tau": 12.0,   # timescale separation (larger -> slower recovery variable)
    "I0": 0.50,    # constant drive

    # --- Mapping to biochemical abundances (molecules / arbitrary concentration units) ---
    # Offsets ensure positivity even though u,v swing around zero.
    "A_scale": 120.0,
    "I_scale": 120.0,
    "u0": 2.50,
    "v0": 2.50,

    # --- Metabolic context ---
    # N_pool: buffered reservoir (chemostat-like)
    "N0": 780.0,
    "kN_relax": 0.0020,  # relaxation back to N0 (buffer strength)
    "kN_cons": 0.0500,   # A-dependent consumption strength
    "K_N": 400.0,        # saturation in N
    "K_A": 200.0,        # saturation in A

    # P_product: produced from A & N (saturable), decays slowly
    "kP_prod": 0.080,
    "dP": 0.05,
}

# Initial conditions in (u, v, N, P) coordinates
# (u,v) chosen off the fixed point so it converges to the limit cycle.
Y0 = [-1.0, 1.0, PARAMS["N0"], 10.0]

TSPAN = (0.0, 600.0)

def _sat(x, K):
    # Simple saturation x/(K+x), numerically safe for x>=0
    x = max(0.0, float(x))
    return x / (K + x)

def dYdt(t, y):
    p = PARAMS
    u, v, N, P = y

    # --- FitzHugh–Nagumo core (relaxation oscillator) ---
    du = u - (u**3) / 3.0 - v + p["I0"]
    dv = (u + p["a"] - p["b"] * v) / p["tau"]

    # Map to biochemical abundances (positive by design via offsets)
    A = p["A_scale"] * (u + p["u0"])
    I = p["I_scale"] * (v + p["v0"])

    # --- Metabolic context (buffered pool + saturable consumption/production) ---
    # Nutrient relaxes to setpoint N0, but is consumed when A is high.
    v_relax = p["kN_relax"] * (p["N0"] - N)
    v_cons = p["kN_cons"] * (_sat(A, p["K_A"])) * (_sat(N, p["K_N"])) * N
    dN = v_relax - v_cons

    # Product is generated from nutrient flux, decays slowly.
    v_prod = p["kP_prod"] * (_sat(A, p["K_A"])) * (_sat(N, p["K_N"])) * N
    dP = v_prod - p["dP"] * max(P, 0.0)

    return np.array([du, dv, dN, dP], dtype=float)

def _to_observables(sol):
    """Convert solution in (u,v,N,P) to observables (A,I,N,P)."""
    p = PARAMS
    u = sol.y[0]
    v = sol.y[1]
    A = p["A_scale"] * (u + p["u0"])
    I = p["I_scale"] * (v + p["v0"])
    N = sol.y[2]
    P = sol.y[3]
    return np.vstack([A, I, N, P])

def _audit_labels(y_obs):
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        series = y_obs[i]
        q80 = np.quantile(series, 0.80)
        q99 = np.quantile(series, 0.99)
        label = int((q80 < 200.0) and (q99 < 200.0))
        labels.append(label)
        print(f"{name}: q80={q80:.3f}, q99={q99:.3f} => label={label}")
    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass model (may not be useful for training)")

if __name__ == "__main__":
    t_eval = np.linspace(TSPAN[0], TSPAN[1], 4000)
    sol = solve_ivp(
        dYdt,
        TSPAN,
        Y0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )

    y_obs = _to_observables(sol)

    # --- Console audit (q80/q99 labels) ---
    _audit_labels(y_obs)

    # Preview plot of species trajectories
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Regulatory variables (A, I)
    ax1.plot(sol.t, y_obs[0], label="A_activator")
    ax1.plot(sol.t, y_obs[1], label="I_inhibitor")
    ax1.set_ylabel("abundance / concentration")
    ax1.set_title(f"{MODEL_NAME} (deterministic) — limit-cycle regime")
    ax1.legend(loc="best")

    # Metabolic context (N, P)
    ax2.plot(sol.t, y_obs[2], label="N_pool")
    ax2.plot(sol.t, y_obs[3], label="P_product")
    ax2.set_xlabel("t")
    ax2.set_ylabel("abundance / concentration")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig("stochosc04_fitzhugh_nagumo_like_preview.png", dpi=220)

def model_odes(t, y):
    return dYdt(t, y)