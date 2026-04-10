# ============================================================================
# Family 4 (Deterministic Oscillators) — Extended Rosenzweig–MacArthur predator–prey model
# osc02_lotka_volterra_logistic.py
# Logistic prey growth with Holling type II predation, coupled to auxiliary
# bounded nutrient, waste, and low-copy control variables.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "osc02_lotka_volterra_logistic"

SPECIES_NAMES = [
    "Prey",     # S0: prey/resource population (high copy)
    "Pred",     # S1: predator/consumer population (high copy)
    "Nutrient", # S2: bounded environmental driver (high copy; does NOT ramp)
    "Waste",    # S3: byproduct pool (high copy; weak clearance)
    "Ctrl",     # S4: low-copy control state (mildly modulates attack rate; non-quenching)
]

# ===== PARAMETERS ============================================================
# Core RM:
#   dP = r P (1 - P/K) - (a P Z)/(1 + a h P)
#   dZ = e (a P Z)/(1 + a h P) - m Z

PARAMS = {
    # Rosenzweig–MacArthur core
    "r": 0.030,        # /min prey intrinsic growth
    "K": 20000.0,      # molecules carrying capacity
    "a": 5.0e-6,       # 1/(molecule*min) attack rate
    "h": 10.0,         # min handling time (Holling-II saturation)
    "e": 0.60,         # conversion efficiency
    "m": 0.0040,       # /min predator mortality

    # Auxiliary bounded nutrient relaxing to a fixed baseline
    "N0": 12000.0,     # molecules
    "kN": 0.010,       # /min

    # Waste (weak sink; no over-strong removal)
    "kW_prod": 0.06,   # molecules/min per (predation flux unit)
    "kW_clear": 0.0012,# /min

    # Low-copy controller: slow relaxation to baseline; mild effect on attack rate
    "C0": 45.0,        # molecules
    "kC": 0.012,       # /min
    "Kc": 70.0,        # molecules
    "nc": 2.0,         # Hill exponent (mild)
    "a_gain": 0.20,    # max fractional increase of 'a' from Ctrl (kept small)
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    12000.0,  # Prey
    1200.0,   # Pred
    12000.0,  # Nutrient
    600.0,    # Waste
    45.0,     # Ctrl
]

TSPAN = (0.0, 3000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    P, Z, N, W, C = Y

    # Bounded nutrient (prevents monotone ramp artifacts)
    dN = p["kN"] * (p["N0"] - N)

    # Low-copy controller relaxation
    dC = p["kC"] * (p["C0"] - C)

    # Mild control modulation of the attack rate
    c_act = (C ** p["nc"]) / (p["Kc"] ** p["nc"] + C ** p["nc"])
    a_eff = p["a"] * (1.0 + p["a_gain"] * c_act)

    # Holling type II functional response (per predator)
    f = (a_eff * P) / (1.0 + a_eff * p["h"] * P)

    # Consumption flux
    v_consume = f * Z

    # RM core dynamics
    dP = p["r"] * P * (1.0 - P / p["K"]) - v_consume
    dZ = p["e"] * v_consume - p["m"] * Z

    # Waste: produced by consumption, weak clearance 
    dW = p["kW_prod"] * v_consume - p["kW_clear"] * W

    return np.array([dP, dZ, dN, dW, dC], dtype=float)

# ===== SELF-TEST / MICRO-AUDIT ==============================================
if __name__ == "__main__":
    t_span = TSPAN
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    # Micro-audit: per-species q80/q99 labels
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
    plt.title("osc02_lotka_volterra_logistic (Holling-II predator–prey oscillator)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("osc02_lotka_volterra_logistic_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
