# ============================================================================
# Family 1 (Metabolism) — Linear pathway with homeostatic feedback 
# metab03_linear_pathway_feedback.py
# Linear metabolic pathway with a regulated committed step, product inhibition
# by pyruvate, and a slow integral-feedback regulator that adjusts enzyme
# abundance to maintain pyruvate near a reference level.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "metab03_linear_pathway_feedback"

SPECIES_NAMES = [
    "Glc_ext",
    "G6P",
    "F6P",
    "FBP",
    "GAP",
    "Pyr",
    "E_lim",
    "Reg_inhib",
]

PARAMS = {
    "J_in": 9500.0,
    "k_out_pyr": 0.020,

    "k1": 0.060,
    "k2": 0.045,
    "k4": 0.030,
    "k5": 0.026,
    "k2_leak": 0.010,

    # Committed step (E-limited, Pyr inhibited)
    "kcat3": 1.9,
    "Km3": 1100.0,
    "Ki_Pyr": 9000.0,
    "n_Pyr": 2.0,

    # Enzyme synthesis repressed by Reg_inhib
    "alpha_E": 1.1,
    "K_regE": 40.0,
    "n_regE": 2.0,
    "delta_E": 0.012,

    # Integral feedback / homeostatic control on Pyr
    "Pyr_ref": 1200.0,   # setpoint (close to initial Pyr)
    "k_int": 0.0015,     # integral gain (/min)
    "delta_R": 0.008,    # slow relaxation (/min)
    "R_min": 0.0,        # soft lower reference
    "R_max": 200.0,      # used only for soft effect via repression saturation
}

Y0 = [
    2400.0,
    550.0,
    380.0,
    200.0,
    170.0,
    1200.0,
    65.0,
    20.0,
]

TSPAN = (0.0, 2000.0)

def dYdt(t, Y):
    p = PARAMS
    Glc, G6P, F6P, FBP, GAP, Pyr, E, R = Y

    v1 = p["k1"] * Glc
    v2 = p["k2"] * G6P

    inhib_Pyr = 1.0 / (1.0 + (Pyr / p["Ki_Pyr"]) ** p["n_Pyr"])
    v3 = inhib_Pyr * p["kcat3"] * E * (F6P / (p["Km3"] + F6P))

    v4 = p["k4"] * FBP
    v5 = p["k5"] * GAP

    dGlc = p["J_in"] - v1
    dG6P = v1 - v2
    dF6P = v2 - v3 - p["k2_leak"] * F6P
    dFBP = v3 - v4
    dGAP = v4 - v5
    dPyr = v5 - p["k_out_pyr"] * Pyr

    # Integral feedback regulator:
    # R integrates (Pyr - Pyr_ref) with slow leak to keep bounded.
    dR = p["k_int"] * (Pyr - p["Pyr_ref"]) - p["delta_R"] * (R - p["R_min"])

    # Enzyme synthesis repressed by R 
    R_eff = max(R, 0.0)
    repress_E = 1.0 / (1.0 + (R_eff / p["K_regE"]) ** p["n_regE"])
    synth_E = p["alpha_E"] * repress_E
    dE = synth_E - p["delta_E"] * E

    return np.array([dGlc, dG6P, dF6P, dFBP, dGAP, dPyr, dE, dR], dtype=float)

def model_odes(t, y):
    return dYdt(t, y)

if __name__ == "__main__":
    t_eval = np.linspace(TSPAN[0], TSPAN[1], 2000)
    sol = solve_ivp(dYdt, TSPAN, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        y = sol.y[i]
        q80 = np.quantile(y, 0.80)
        q99 = np.quantile(y, 0.99)
        label = int((q80 < 200) and (q99 < 200))
        labels.append(label)
        print(f"{name}: q80={q80:.2f}, q99={q99:.2f}, label={label}")

    if len(set(labels)) == 1:
        print("WARNING: monoclass model")

    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("metab03 (homeostatic integral feedback on Pyr)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metab03_linear_pathway_feedback_preview.png", dpi=220)
