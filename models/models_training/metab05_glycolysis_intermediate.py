# metab05_glycolysis_intermediate_v2.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "metab05_glycolysis_intermediate"

SPECIES_NAMES = [
    "Glc_ext",
    "G6P",
    "F6P",
    "FBP",
    "GAP",
    "Pyr",
    "ATP",
    "ADP",
    "E_PFK",
    "FBP_Buf",
]

PARAMS = {
    # Glucose supply and usage
    "J_in": 11000.0,
    "k_glc_use": 0.055,

    # Hexokinase (ATP-dependent)
    "k_hk": 0.060,
    "K_ATP_hk": 1400.0,

    # Isomerization
    "k_iso_fwd": 0.038,
    "k_iso_rev": 0.018,

    # PFK (energy charge regulation)
    "kcat_pfk": 2.0,
    "Km_pfk": 950.0,
    "K_inhib_ATP": 4200.0,
    "n_inhib_ATP": 2.0,
    "K_act_ADP": 1100.0,
    "n_act_ADP": 2.0,

    # FBP Buffering
    "Buf_tot": 90.0,
    "k_on_buf": 2.5e-5,
    "k_off_buf": 0.030,

    # Aldolase
    "k_ald": 0.028,

    # GAP -> Pyr baseline
    "k_gap": 0.022,

    # NEW FEED-FORWARD ACTIVATION (FBP → activates PK)
    "K_FBP_PK": 800.0,
    "n_FBP_PK": 2.0,

    # ATP regeneration
    "k_atp_regen": 0.038,
    "K_ADP_regen": 1100.0,

    # Pyr clearance
    "k_pyr_out": 0.018,

    # ATPase
    "k_atpase": 0.020,

    # Enzyme turnover
    "alpha_E": 0.85,
    "delta_E": 0.012,
}

Y0 = [
    3200.0,
    650.0,
    520.0,
    260.0,
    240.0,
    850.0,
    11500.0,
    3200.0,
    65.0,
    10.0,
]

TSPAN = (0.0, 2000.0)

def dYdt(t, Y):
    p = PARAMS
    Glc, G6P, F6P, FBP, GAP, Pyr, ATP, ADP, E, FBP_Buf = Y

    Buf_free = max(p["Buf_tot"] - FBP_Buf, 0.0)

    v_glc_in = p["J_in"]
    v_glc_use = p["k_glc_use"] * Glc

    hk_ATP = ATP / (p["K_ATP_hk"] + ATP)
    v_hk = p["k_hk"] * Glc * hk_ATP

    v_iso_fwd = p["k_iso_fwd"] * G6P
    v_iso_rev = p["k_iso_rev"] * F6P

    inhib_ATP = 1.0 / (1.0 + (ATP / p["K_inhib_ATP"]) ** p["n_inhib_ATP"])
    act_ADP  = (ADP**p["n_act_ADP"]) / (p["K_act_ADP"]**p["n_act_ADP"] + ADP**p["n_act_ADP"])
    v_pfk = inhib_ATP * (0.2 + 0.8 * act_ADP) * p["kcat_pfk"] * E * (F6P / (p["Km_pfk"] + F6P))

    # Buffering
    v_buf_on  = p["k_on_buf"]  * FBP * Buf_free
    v_buf_off = p["k_off_buf"] * FBP_Buf

    v_ald = p["k_ald"] * FBP

    # NEW FEED-FORWARD activation of GAP→Pyr by FBP
    act_FBP_PK = (FBP**p["n_FBP_PK"]) / (p["K_FBP_PK"]**p["n_FBP_PK"] + FBP**p["n_FBP_PK"])
    v_gap = p["k_gap"] * GAP * (0.3 + 0.7 * act_FBP_PK)

    regen_ADP = ADP / (p["K_ADP_regen"] + ADP)
    v_atp_regen = p["k_atp_regen"] * Pyr * regen_ADP

    v_pyr_out = p["k_pyr_out"] * Pyr
    v_atpase  = p["k_atpase"] * ATP

    dE = p["alpha_E"] - p["delta_E"] * E

    dGlc = v_glc_in - v_glc_use - v_hk
    dG6P = v_hk - v_iso_fwd + v_iso_rev
    dF6P = v_iso_fwd - v_iso_rev - v_pfk
    dFBP = v_pfk - v_ald - v_buf_on + v_buf_off
    dFBP_Buf = v_buf_on - v_buf_off
    dGAP = 2.0 * v_ald - v_gap
    dPyr = v_gap - v_atp_regen - v_pyr_out

    dATP = -v_hk - v_pfk - v_atpase + v_atp_regen
    dADP =  v_hk + v_pfk + v_atpase - v_atp_regen

    return np.array([dGlc, dG6P, dF6P, dFBP, dGAP, dPyr, dATP, dADP, dE, dFBP_Buf], dtype=float)

def model_odes(t, y):
    return dYdt(t, y)

if __name__ == "__main__":
    t_eval = np.linspace(TSPAN[0], TSPAN[1], 2000)
    sol = solve_ivp(dYdt, TSPAN, Y0, t_eval=t_eval, method="Radau", atol=1e-9, rtol=1e-7)

    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        y = sol.y[i]
        q80 = np.quantile(y, 0.80)
        q99 = np.quantile(y, 0.99)
        label = int((q80 < 200) and (q99 < 200))
        print(f"{name}: q80={q80:.2f}, q99={q99:.2f}, label={label}")
        labels.append(label)

    print(f"Label0={labels.count(0)}, Label1={labels.count(1)}")
    if len(set(labels)) == 1:
        print("WARNING: monoclass model")

    plt.figure(figsize=(10,5))
    for i in range(len(SPECIES_NAMES)):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.legend()
    plt.title("metab05_glycolysis_intermediate_v2 (FBP→PK feed-forward)")
    plt.tight_layout()
    plt.savefig("metab05_v2_preview.png", dpi=200)
