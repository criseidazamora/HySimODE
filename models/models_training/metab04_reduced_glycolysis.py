# metab04_reduced_glycolysis.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 1 (Metabolism) — Reduced glycolysis with a futile cycle (PFK <-> FBPase)
# ============================================================================
MODEL_NAME = "metab04_reduced_glycolysis"

SPECIES_NAMES = [
    "Glc_ext",  # S0
    "G6P",      # S1
    "F6P",      # S2
    "FBP",      # S3
    "GAP",      # S4
    "Pyr",      # S5
    "ATP",      # S6
    "ADP",      # S7
    "E_PFK",    # S8
]

# ===== PARAMETERS ============================================================
PARAMS = {
    # Glucose supply and sink
    "J_in": 12000.0,     # glucose influx (molecules/min)
    "k_glc_use": 0.060,  # effective uptake/processing of Glc_ext (/min)

    # Hexokinase-like: Glc -> G6P with ATP dependence (saturable in ATP)
    "k_hk": 0.065,       # (/min)
    "K_ATP_hk": 1500.0,  # (molecules)

    # Isomerization: G6P <-> F6P (reversible mass-action)
    "k_iso_fwd": 0.040,  # (/min)
    "k_iso_rev": 0.020,  # (/min)

    # PFK committed step: F6P -> FBP (enzyme-catalyzed, saturable; inhibited by ATP, activated by ADP)
    "kcat_pfk": 2.2,        # (/min)
    "Km_pfk": 1000.0,       # (molecules)
    "K_inhib_ATP": 4000.0,  # (molecules)
    "n_inhib_ATP": 2.0,     # Hill coefficient
    "K_act_ADP": 1200.0,    # (molecules)
    "n_act_ADP": 2.0,       # Hill coefficient

    # FUTILE CYCLE: FBPase-like reverse flux: FBP -> F6P (saturable drain)
    # Biological interpretation: partial gluconeogenic activity / futile cycling control.
    "k_fbpase": 0.030,      # (/min)
    "K_fbpase": 800.0,      # (molecules)

    # ATP cost coupled to futile cycle activity (represents energetic penalty of cycling)
    # Kept modest to maintain stability but enough to change ATP/ADP statistics.
    "k_cycle_atp": 0.015,   # (/min)
    "K_cycle_atp": 600.0,   # (molecules)

    # Aldolase-like: FBP -> 2*GAP (mass-action)
    "k_ald": 0.030,         # (/min)

    # Lower glycolysis lumped: GAP -> Pyr (mass-action)
    "k_gap": 0.022,         # (/min)

    # ATP regeneration from Pyr (substrate-level phosphorylation, saturable in ADP)
    "k_atp_regen": 0.040,   # (/min)
    "K_ADP_regen": 1200.0,  # (molecules)

    # Pyruvate clearance
    "k_pyr_out": 0.018,     # (/min)

    # Basal ATPase load: ATP -> ADP
    "k_atpase": 0.020,      # (/min)

    # Enzyme turnover (low-copy)
    "alpha_E": 0.9,         # synthesis (molecules/min)
    "delta_E": 0.012,       # dilution/degradation (/min)
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    3000.0,  # Glc_ext
    600.0,   # G6P
    500.0,   # F6P
    250.0,   # FBP
    250.0,   # GAP
    800.0,   # Pyr
    12000.0, # ATP
    3000.0,  # ADP
    70.0,    # E_PFK
]

TSPAN = (0.0, 2000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Glc, G6P, F6P, FBP, GAP, Pyr, ATP, ADP, E = Y

    # Glucose influx and sink
    v_glc_in = p["J_in"]
    v_glc_use = p["k_glc_use"] * Glc

    # Hexokinase-like phosphorylation (ATP-dependent)
    hk_ATP = ATP / (p["K_ATP_hk"] + ATP)
    v_hk = p["k_hk"] * Glc * hk_ATP

    # Isomerization reversible
    v_iso_fwd = p["k_iso_fwd"] * G6P
    v_iso_rev = p["k_iso_rev"] * F6P

    # PFK step with energy charge regulation
    inhib_ATP = 1.0 / (1.0 + (ATP / p["K_inhib_ATP"]) ** p["n_inhib_ATP"])
    act_ADP = (ADP ** p["n_act_ADP"]) / (p["K_act_ADP"] ** p["n_act_ADP"] + ADP ** p["n_act_ADP"])
    v_pfk = inhib_ATP * (0.2 + 0.8 * act_ADP) * p["kcat_pfk"] * E * (F6P / (p["Km_pfk"] + F6P))

    # FUTILE CYCLE: FBPase reverse flux back to F6P
    v_fbpase = p["k_fbpase"] * FBP / (p["K_fbpase"] + FBP)

    # ATP energetic cost coupled to futile cycling (modest, saturable in FBP)
    v_cycle_atp = p["k_cycle_atp"] * FBP / (p["K_cycle_atp"] + FBP)

    # Aldolase: FBP -> 2*GAP
    v_ald = p["k_ald"] * FBP

    # Lower glycolysis: GAP -> Pyr
    v_gap = p["k_gap"] * GAP

    # ATP regeneration from Pyr (requires ADP)
    regen_ADP = ADP / (p["K_ADP_regen"] + ADP)
    v_atp_regen = p["k_atp_regen"] * Pyr * regen_ADP

    # Pyruvate outflow
    v_pyr_out = p["k_pyr_out"] * Pyr

    # Basal ATPase load
    v_atpase = p["k_atpase"] * ATP

    # Enzyme turnover
    dE = p["alpha_E"] - p["delta_E"] * E

    # Mass balances
    dGlc = v_glc_in - v_glc_use - v_hk
    dG6P = v_hk - v_iso_fwd + v_iso_rev

    # Futile cycle adds +v_fbpase to F6P and -v_fbpase to FBP
    dF6P = v_iso_fwd - v_iso_rev - v_pfk + v_fbpase
    dFBP = v_pfk - v_ald - v_fbpase

    dGAP = 2.0 * v_ald - v_gap
    dPyr = v_gap - v_atp_regen - v_pyr_out

    # Nucleotide interconversion:
    # ATP consumed by HK + PFK + ATPase + cycling cost; regenerated from Pyr.
    dATP = -v_hk - v_pfk - v_atpase - v_cycle_atp + v_atp_regen
    dADP =  v_hk + v_pfk + v_atpase + v_cycle_atp - v_atp_regen

    return np.array([dGlc, dG6P, dF6P, dFBP, dGAP, dPyr, dATP, dADP, dE], dtype=float)

# ===== SELF-TEST / MICRO-AUDIT ==============================================
if __name__ == "__main__":
    t_span = TSPAN
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

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

    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("metab04_reduced_glycolysis v3 (PFK <-> FBPase futile cycle + ATP cost)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metab04_reduced_glycolysis_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
