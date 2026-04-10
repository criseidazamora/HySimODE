# ============================================================================
# Family 5 (Gene Regulatory Circuits) — Repressilator with dimeric repressors
# grn03_repressilator_dimerization.py
# Three genes in a ring: A represses B, B represses C, C represses A
# Repression is mediated by protein dimers (explicit dimer species).
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "grn03_repressilator_dimerization"

SPECIES_NAMES = [
    "mA", "mB", "mC",      # mRNAs (low copy)
    "pA", "pB", "pC",      # monomeric proteins (higher copy)
    "dA", "dB", "dC",      # dimers (regulatory repressors; often lower than monomers)
]

PARAMS = {
    # Transcription (repressed by upstream dimer)
    "alpha_max": 18.0,    # molecules/min
    "alpha_leak": 0.18,   # molecules/min
    "K_rep": 220.0,       # molecules (dimer repression threshold)
    "n_rep": 2.0,         # cooperativity (kept modest; dimerization adds ultrasensitivity)

    # mRNA degradation (linear)
    "delta_m": 0.18,      # /min

    # Translation and protein degradation (linear)
    "k_tl": 6.0,          # molecules/min per mRNA
    "delta_p": 0.010,     # /min

    # Dimerization kinetics (mass-action)
    "k_dim": 2.0e-5,      # 1/(molecule*min)  p + p -> d
    "k_undim": 0.020,     # /min              d -> p + p

    # Dimer degradation (linear; can be slower than monomer)
    "delta_d": 0.006,     # /min
}

Y0 = [
    40.0,  12.0,  55.0,    # mA, mB, mC
    1400.0, 2600.0, 900.0, # pA, pB, pC
    80.0,  140.0,  60.0,   # dA, dB, dC
]

TSPAN = (0.0, 5500.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    mA, mB, mC, pA, pB, pC, dA, dB, dC = Y

    # Repression by dimers in the ring
    repA = 1.0 / (1.0 + (dC / p["K_rep"]) ** p["n_rep"])  # dC represses A
    repB = 1.0 / (1.0 + (dA / p["K_rep"]) ** p["n_rep"])  # dA represses B
    repC = 1.0 / (1.0 + (dB / p["K_rep"]) ** p["n_rep"])  # dB represses C

    v_tx_A = p["alpha_leak"] + p["alpha_max"] * repA
    v_tx_B = p["alpha_leak"] + p["alpha_max"] * repB
    v_tx_C = p["alpha_leak"] + p["alpha_max"] * repC

    dmA = v_tx_A - p["delta_m"] * mA
    dmB = v_tx_B - p["delta_m"] * mB
    dmC = v_tx_C - p["delta_m"] * mC

    # Translation
    v_tl_A = p["k_tl"] * mA
    v_tl_B = p["k_tl"] * mB
    v_tl_C = p["k_tl"] * mC

    # Dimerization / undimerization
    v_dim_A = p["k_dim"] * pA * pA
    v_dim_B = p["k_dim"] * pB * pB
    v_dim_C = p["k_dim"] * pC * pC

    v_undim_A = p["k_undim"] * dA
    v_undim_B = p["k_undim"] * dB
    v_undim_C = p["k_undim"] * dC

    # Monomer dynamics with stoichiometry of two monomers per dimer
    dpA = v_tl_A - 2.0 * v_dim_A + 2.0 * v_undim_A - p["delta_p"] * pA
    dpB = v_tl_B - 2.0 * v_dim_B + 2.0 * v_undim_B - p["delta_p"] * pB
    dpC = v_tl_C - 2.0 * v_dim_C + 2.0 * v_undim_C - p["delta_p"] * pC

    # Dimer dynamics
    ddA = v_dim_A - v_undim_A - p["delta_d"] * dA
    ddB = v_dim_B - v_undim_B - p["delta_d"] * dB
    ddC = v_dim_C - v_undim_C - p["delta_d"] * dC

    return np.array([dmA, dmB, dmC, dpA, dpB, dpC, ddA, ddB, ddC], dtype=float)

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
    plt.title("grn03-repressilator with explicit dimer repressors")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn03_repressilator_dimerization_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
