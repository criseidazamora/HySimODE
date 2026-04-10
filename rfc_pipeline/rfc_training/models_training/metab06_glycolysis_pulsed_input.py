# ============================================================================
# Family 1 (Metabolism) — Pulsed glycolytic pathway 
# metab06_glycolysis_pulsed_input.py
# Reduced metabolic model with periodic substrate input and coupled ATP/ADP
# energy dynamics. The system combines smooth metabolic fluxes with transient
# responses induced by pulsed feeding, yielding non-stationary biochemical
# trajectories with coupled metabolic and energetic dynamics.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "metab06_glycolysis_pulsed_input"

SPECIES_NAMES = [
    "Glc_ext", "G6P", "F6P", "FBP", "GAP", "Pyr", "ATP", "ADP", "E"
]

PARAMS = {
    # Periodic feeding / perfusion (fed-batch / bolus feeding)
    "J_base": 6000.0,
    "J_amp": 14000.0,
    "pulse_period": 250.0,   # min
    "pulse_duty": 0.25,      # fraction of period "ON"
    "pulse_smooth": 5.0,     # min (smooth edges)

    # Basal glucose usage / clearance (non-modeled consumption)
    "k_glc_use": 0.10,

    # Hexokinase-like (HK): Glc -> G6P, requires ATP, inhibited by G6P (product inhibition)
    "k_hk": 0.055,
    "K_ATP_hk": 1300.0,
    "K_G6P_hk": 400.0,       # product inhibition scale (plausible)
    "n_G6P_hk": 2.0,         # cooperativity for inhibition

    # Isomerase reversible: G6P <-> F6P
    "k_iso_fwd": 0.030,
    "k_iso_rev": 0.018,

    # PFK-like (ATP inhibition + ADP activation)
    "k_pfk": 0.060,
    "K_F6P_pfk": 1800.0,
    "K_inhib_ATP": 4000.0,
    "n_inhib_ATP": 2.0,
    "K_act_ADP": 1200.0,
    "n_act_ADP": 2.0,

    # Aldolase & downstream lumping
    "k_ald": 0.029,
    "k_gap": 0.022,

    # Shunt from G6P to alternative pathway (PPP / storage), saturable drain
    "k_shunt": 0.020,
    "K_G6P_shunt": 900.0,

    # ATP regeneration (e.g., oxidative / substrate-level lumped), depends on Pyr and ADP
    "k_atp_regen": 0.040,
    "K_ADP_regen": 1100.0,

    # Pyruvate clearance & ATPase
    "k_pyr_out": 0.018,
    "k_atpase": 0.020,

    # Enzyme pool turnover
    "alpha_E": 12.0,
    "delta_E": 0.0025,
}

Y0 = np.array([
    20000.0,  # Glc_ext
    200.0,    # G6P
    120.0,    # F6P
    80.0,     # FBP
    60.0,     # GAP
    90.0,     # Pyr
    7000.0,   # ATP
    1500.0,   # ADP
    900.0,    # E
], dtype=float)

TSPAN = (0.0, 1200.0)

# -----------------------------------------------------------------------------
# Core RHS: rhs(t, y, p)
# -----------------------------------------------------------------------------
def rhs(t, Y, p):
    Glc, G6P, F6P, FBP, GAP, Pyr, ATP, ADP, E = Y

    # Periodic feeding/perfusion with smooth edges (tanh)
    P = p["pulse_period"]
    duty = p["pulse_duty"]
    tau = max(float(p["pulse_smooth"]), 1e-9)
    t_in_period = np.mod(t, P)

    def s(x):
        return 0.5 * (1.0 + np.tanh(x / tau))

    t_on = 0.0
    t_off = duty * P
    pulse = s(t_in_period - t_on) * (1.0 - s(t_in_period - t_off))
    J_in = p["J_base"] + p["J_amp"] * pulse

    # Basal glucose clearance/usage
    v_glc_use = p["k_glc_use"] * Glc

    # HK-like with ATP requirement + G6P product inhibition
    hk_ATP = ATP / (p["K_ATP_hk"] + ATP)
    inhib_G6P = 1.0 / (1.0 + (G6P / p["K_G6P_hk"]) ** p["n_G6P_hk"])
    v_hk = p["k_hk"] * Glc * hk_ATP * inhib_G6P

    # Isomerase reversible
    v_iso_fwd = p["k_iso_fwd"] * G6P
    v_iso_rev = p["k_iso_rev"] * F6P

    # PFK-like: saturación + inhibición ATP + activación ADP (Hill)
    sat_F6P = F6P / (p["K_F6P_pfk"] + F6P)
    inhib_ATP = 1.0 / (1.0 + (ATP / p["K_inhib_ATP"]) ** p["n_inhib_ATP"])
    act_ADP = (ADP ** p["n_act_ADP"]) / (p["K_act_ADP"] ** p["n_act_ADP"] + ADP ** p["n_act_ADP"])
    v_pfk = p["k_pfk"] * E * sat_F6P * inhib_ATP * (0.4 + 0.6 * act_ADP)

    # Aldolase and downstream lumping
    v_ald = p["k_ald"] * FBP
    v_gap = p["k_gap"] * GAP

    # Shunt drain from G6P (PPP/storage)
    v_shunt = p["k_shunt"] * G6P / (p["K_G6P_shunt"] + G6P)

    # ATP regeneration depends on Pyr and ADP
    sat_ADP = ADP / (p["K_ADP_regen"] + ADP)
    v_atp_regen = p["k_atp_regen"] * Pyr * sat_ADP

    # Pyruvate clearance
    v_pyr_out = p["k_pyr_out"] * Pyr

    # ATPase consumption
    v_atpase = p["k_atpase"] * ATP / (1500.0 + ATP)

    # Enzyme turnover
    dE = p["alpha_E"] - p["delta_E"] * E

    # Mass balances
    dGlc = J_in - v_glc_use - v_hk
    dG6P = v_hk - v_iso_fwd + v_iso_rev - v_shunt
    dF6P = v_iso_fwd - v_iso_rev - v_pfk
    dFBP = v_pfk - v_ald
    dGAP = 2.0 * v_ald - v_gap
    dPyr = v_gap - v_atp_regen - v_pyr_out

    dATP = -v_hk - v_pfk - v_atpase + v_atp_regen
    dADP =  v_hk + v_pfk + v_atpase - v_atp_regen

    return np.array([dGlc, dG6P, dF6P, dFBP, dGAP, dPyr, dATP, dADP, dE], dtype=float)

# -----------------------------------------------------------------------------
# Compatibility wrapper for the model RHS
# -----------------------------------------------------------------------------
def dYdt(t, Y, p=None):
    if p is None:
        p = PARAMS
    return rhs(t, Y, p)

# -----------------------------------------------------------------------------
# ODE interface required by the audit pipeline
# -----------------------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return rhs(t, y, PARAMS)

# -----------------------------------------------------------------------------
# Main simulation and micro-audit
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    t0, tf = TSPAN
    t_eval = np.linspace(t0, tf, 2000)

    sol = solve_ivp(
        fun=lambda t, y: model_odes(t, y),
        t_span=(t0, tf),
        y0=Y0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    Y = sol.y
    labels = []

    print(f"MODEL: {MODEL_NAME}")
    for i, name in enumerate(SPECIES_NAMES):
        y = Y[i, :]
        q80 = float(np.quantile(y, 0.80))
        q99 = float(np.quantile(y, 0.99))
        label = int((q80 < 200) and (q99 < 200))
        labels.append(label)
        print(f"{name:>8s} | q80={q80:10.3f}  q99={q99:10.3f}  label={label}")

    if len(set(labels)) == 1:
        print("WARNING: monoclass labels in this model (all species same label).")

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(SPECIES_NAMES):
        plt.plot(sol.t, Y[i, :], label=name, linewidth=1.2)
    plt.title("metab06_glycolysis_pulsed_input (pulsed substrate input)")
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
  
