# ============================================================================
# Family 1 (Metabolism) — Linear pathway with one Michaelis–Menten step 
# metab02_linear_pathway_MM.py
# Linear metabolic chain with a single enzyme-limited saturable conversion
# (F6P→FBP), plus a slow product-activated regulator that introduces mild
# negative feedback on catalytic throughput.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "metab02_linear_pathway_MM"

SPECIES_NAMES = [
    "Glc_ext",      # S0: input substrate pool (high copy)
    "G6P",          # S1: intermediate (high copy)
    "F6P",          # S2: intermediate (high copy)
    "FBP",          # S3: intermediate (high copy)
    "GAP",          # S4: intermediate (high copy)
    "Pyr",          # S5: end product (high copy)
    "E_lim",        # S6: limiting enzyme pool (low copy)
    "Reg_feedback", # S7: product-activated regulator (low copy)
]

# ===== PARAMETERS ============================================================
# Units:
#  - fluxes: molecules/min
#  - first-order rates: /min
#  - MM parameters: kcat (/min), Km (molecules)
PARAMS = {
    # Input / outflow
    "J_in": 9000.0,     # constant substrate influx (molecules/min)
    "k_out_pyr": 0.018, # product clearance (/min)

    # Linear chain steps (mass-action)
    "k1": 0.060,   # Glc_ext -> G6P
    "k2": 0.045,   # G6P -> F6P
    "k4": 0.030,   # FBP -> GAP
    "k5": 0.026,   # GAP -> Pyr
    "k2_leak": 0.012,  # F6P side loss (/min)

    # Saturable step (MM approximation for E_lim-catalyzed F6P -> FBP)
    "kcat3": 1.8,     # turnover (/min)
    "Km3": 1200.0,    # Michaelis constant (molecules)

    # Enzyme turnover (low-copy)
    "alpha_E": 0.9,    # synthesis (molecules/min)
    "delta_E": 0.011,  # degradation/dilution (/min)

    # Product-activated regulator turnover (low-copy)
    "alpha_R0": 0.5,     # basal synthesis (molecules/min)
    "k_act_R": 0.9,      # induced synthesis amplitude (molecules/min)
    "K_act_R": 9000.0,   # activation scale (molecules)
    "delta_R": 0.020,    # degradation (/min)

    # Regulator-mediated mild inhibition of catalytic throughput
    "K_inhib": 140.0,    # inhibition scale (molecules)
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    2500.0,  # Glc_ext
    600.0,   # G6P
    400.0,   # F6P
    220.0,   # FBP
    180.0,   # GAP
    900.0,   # Pyr
    55.0,    # E_lim
    25.0,    # Reg_feedback
]

TSPAN = (0.0, 2000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Glc, G6P, F6P, FBP, GAP, Pyr, E, R = Y

    # Mild saturating inhibition (dimensionless)
    inhib = 1.0 / (1.0 + (R / p["K_inhib"]))

    v1 = p["k1"] * Glc
    v2 = p["k2"] * G6P

    # Michaelis–Menten step with explicit enzyme pool:
    # v3 = kcat * E * F6P / (Km + F6P), optionally inhibited by regulator
    v3 = inhib * p["kcat3"] * E * (F6P / (p["Km3"] + F6P))

    v4 = p["k4"] * FBP
    v5 = p["k5"] * GAP

    # Product-activated regulator synthesis (saturating)
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

    # Optional plot 
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("metab02_linear_pathway_MM (one saturable enzyme-catalyzed step)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metab02_linear_pathway_MM_preview.png", dpi=220)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

# ODE interface required by the audit pipeline
def model_odes(t, y):
    return dYdt(t, y)

