# ============================================================================
# Family 1 (Metabolism) — Linear pathway with low-copy enzyme regulation
# metab01_linear_pathway.py
# Linear metabolic chain with mass-action kinetics, a limiting enzyme-mediated
# committed step, and a slow product-activated regulator that mildly represses
# catalytic throughput.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "metab01_linear_pathway"

SPECIES_NAMES = [
    "Glc_ext",      # S0: input substrate pool (high copy)
    "G6P",          # S1: intermediate (high copy)
    "F6P",          # S2: intermediate (high copy)
    "FBP",          # S3: intermediate (high copy)
    "GAP",          # S4: intermediate (high copy)
    "Pyr",          # S5: end product (high copy)
    "E_lim",        # S6: limiting enzyme pool (low copy)
    "Reg_feedback", # S7: regulator activated by product (low copy)
]

# ===== PARAMETERS ============================================================
# Units:
#  - fluxes: molecules/min
#  - first-order rates: /min
#  - second-order-like catalytic rates: 1/(molecule*min) in copy-number form
PARAMS = {
    # Input / outflow
    "J_in": 8000.0,     # constant substrate influx (molecules/min)
    "k_out_pyr": 0.020, # product clearance (/min)

    # Linear chain steps (mass-action)
    "k1": 0.060,   # Glc_ext -> G6P
    "k2": 0.045,   # G6P -> F6P
    "k4": 0.030,   # FBP -> GAP
    "k5": 0.025,   # GAP -> Pyr

    # Enzyme-catalyzed committed step with mass-action dependence on E_lim and F6P
    "k3_cat": 3.0e-5,   # catalytic conversion rate (1/(molecule*min))
    "k2_leak": 0.010,   # F6P side loss (/min) to represent competing pathways

    # Enzyme turnover (low-copy, plausible regulator/enzyme scale)
    "alpha_E": 1.2,     # E_lim synthesis (molecules/min)
    "delta_E": 0.012,   # E_lim degradation/dilution (/min)

    # Feedback regulator turnover (low-copy controller, product-activated synthesis)
    "alpha_R0": 0.6,    # basal Reg synthesis (molecules/min)
    "k_act_R": 1.5,     # induced synthesis amplitude (molecules/min)
    "K_act_R": 4000.0,  # activation scale (molecules)
    "delta_R": 0.020,   # Reg degradation (/min)

    # Regulator-mediated modulation of effective catalysis through mild inhibition
    # Product-responsive regulator reduces effective catalytic throughput
    "K_inhib": 120.0,   # inhibition scale (molecules)
}

# ===== INITIAL CONDITIONS ====================================================
# 8 species total: metabolites high-copy; enzyme/regulator low-copy
Y0 = [
    2000.0,  # Glc_ext
    500.0,   # G6P
    300.0,   # F6P
    200.0,   # FBP
    150.0,   # GAP
    800.0,   # Pyr
    60.0,    # E_lim
    30.0,    # Reg_feedback
]

TSPAN = (0.0, 2000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Glc, G6P, F6P, FBP, GAP, Pyr, E, R = Y

    # Effective inhibition of enzyme throughput by regulator (mild, saturating)
    inhib = 1.0 / (1.0 + (R / p["K_inhib"]))

    v1 = p["k1"] * Glc
    v2 = p["k2"] * G6P

    # Enzyme-catalyzed step (mass-action with enzyme + substrate; moderated by regulator)
    v3 = p["k3_cat"] * inhib * E * F6P

    v4 = p["k4"] * FBP
    v5 = p["k5"] * GAP

    # Regulator activation by product (saturating induced synthesis)
    vR_act = p["k_act_R"] * Pyr / (p["K_act_R"] + Pyr)

    dGlc = p["J_in"] - v1
    dG6P = v1 - v2
    dF6P = v2 - v3 - p["k2_leak"] * F6P
    dFBP = v3 - v4
    dGAP = v4 - v5
    dPyr = v5 - p["k_out_pyr"] * Pyr

    dE = p["alpha_E"] - p["delta_E"] * E
    dR = (p["alpha_R0"] + vR_act) - p["delta_R"] * R

    return np.array([dGlc, dG6P, dF6P, dFBP, dGAP, dPyr, dE, dR], dtype=float)

# ===== MICRO-AUDIT ==============================================
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
    plt.title("metab01_linear_pathway (mass-action chain with low-copy enzyme/regulator)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metab01_linear_pathway_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
