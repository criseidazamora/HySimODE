# ============================================================================
# Family 5 (Gene Regulatory Circuits) — Genetic toggle switch with inducer-modulated repression
# grn04_toggle_switch.py
# Two-gene mutual repression circuit: protein A represses gene B, and protein B
# represses gene A.
# The model includes explicit mRNA/protein layers and slowly relaxing inducer pools
# that reduce repressor activity.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "grn04_toggle_switch"

SPECIES_NAMES = [
    "mA", "mB",   # mRNAs (low copy)
    "pA", "pB",   # proteins (higher copy)
    "IndA",       # inducer affecting A repression activity (high copy driver)
    "IndB",       # inducer affecting B repression activity (high copy driver)
]

PARAMS = {
    # Transcription (mutual Hill repression)
    "alphaA_max": 16.0,   # molecules/min
    "alphaB_max": 16.0,   # molecules/min
    "alpha_leak": 0.15,   # molecules/min (basal leak)

    "K_A": 700.0,         # molecules (pB repression threshold for A)
    "K_B": 700.0,         # molecules (pA repression threshold for B)
    "n_A": 3.0,           # cooperativity
    "n_B": 3.0,

    # mRNA degradation (linear)
    "delta_m": 0.20,      # /min

    # Translation and protein degradation (linear)
    "k_tl": 6.5,          # molecules/min per mRNA
    "delta_p": 0.010,     # /min

    # Inducer pools with bounded relaxation to fixed setpoints
    # Biological interpretation: slowly varying inducer concentrations that reduce repressor activity
    "IndA0": 12000.0,     # molecules
    "IndB0": 6000.0,      # molecules
    "k_ind_relax": 0.0015,# /min

    # Inducer efficacy (reduces effective repression strength via activity scaling)
    "K_ind": 6000.0,      # molecules
}

Y0 = [
    18.0,   # mA
    65.0,   # mB
    2600.0, # pA
    600.0,  # pB
    12000.0,# IndA
    6000.0, # IndB
]

TSPAN = (0.0, 6000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    mA, mB, pA, pB, IndA, IndB = Y

    # Inducers reduce repressor activity (dimensionless factors in [~0.5, 1])
    aA = 1.0 / (1.0 + IndA / p["K_ind"])  # IndA reduces pA activity (repression of B)
    aB = 1.0 / (1.0 + IndB / p["K_ind"])  # IndB reduces pB activity (repression of A)

    # Mutual repression at transcriptional level
    repA = 1.0 / (1.0 + (aB * pB / p["K_A"]) ** p["n_A"])  # pB represses A
    repB = 1.0 / (1.0 + (aA * pA / p["K_B"]) ** p["n_B"])  # pA represses B

    v_tx_A = p["alpha_leak"] + p["alphaA_max"] * repA
    v_tx_B = p["alpha_leak"] + p["alphaB_max"] * repB

    dmA = v_tx_A - p["delta_m"] * mA
    dmB = v_tx_B - p["delta_m"] * mB

    dpA = p["k_tl"] * mA - p["delta_p"] * pA
    dpB = p["k_tl"] * mB - p["delta_p"] * pB

    # Inducer pools relax to setpoints (kept high-copy; adds late-time drift features)
    dIndA = p["k_ind_relax"] * (p["IndA0"] - IndA)
    dIndB = p["k_ind_relax"] * (p["IndB0"] - IndB)

    return np.array([dmA, dmB, dpA, dpB, dIndA, dIndB], dtype=float)

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
    plt.title("grn04 (canonical mutual repression toggle)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn04_toggle_switch_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
