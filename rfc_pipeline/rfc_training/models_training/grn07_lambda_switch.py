# ============================================================================
# Family 5 (Gene Regulatory Circuits) — Lambda phage–type bistable switch
# grn07_lambda_switch.py
# CI–Cro mutual repression with CI positive autoregulation.
# Includes explicit dimerization (active DNA-binding forms) and SOS-like RecA signal
# that drives CI cleavage/inactivation, with a default lysogenic (CI-high) state.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "grn07_lambda_switch"

SPECIES_NAMES = [
    "mCI",    # S0: cI mRNA (low copy)
    "mCro",   # S1: cro mRNA (low copy)
    "CI",     # S2: CI monomer
    "Cro",    # S3: Cro monomer
    "CI2",    # S4: CI dimer (active repressor/activator)
    "Cro2",   # S5: Cro dimer (active repressor)
    "CIc",    # S6: cleaved/inactive CI fragments (damage proxy / sink)
    "RecA",   # S7: SOS/RecA activity signal (basal low; can be raised externally)
]

PARAMS = {
    # Basal and regulated transcription (molecules/min)
    "alpha_CI_max": 14.0,
    "alpha_Cro_max": 16.0,
    "alpha_leak": 0.06,

    # CI positive autoregulation mediated by CI2
    "K_CI_act": 520.0,   # molecules of CI2
    "n_CI_act": 2.0,
    "act_gain": 1.2,     # fold increase above baseline when activated

    # Mutual repression thresholds (dimers)
    "K_CI_rep_Cro": 180.0,  # CI2 represses cro (strong repression)
    "K_Cro_rep_CI": 220.0,  # Cro2 represses cI
    "n_rep": 3.2,

    # mRNA degradation
    "delta_m": 0.22,     # /min

    # Translation (molecules/min per mRNA) and protein degradation
    "k_tl_CI": 5.2,
    "k_tl_Cro": 5.8,
    "delta_CI": 0.010,   # /min
    "delta_Cro": 0.012,  # /min

    # Dimerization (mass-action), undimerization, and dimer turnover
    "k_dim_CI": 2.2e-6,    # 1/(molecule*min)
    "k_undim_CI": 0.060,   # /min
    "k_dim_Cro": 6.0e-6,   # 1/(molecule*min)
    "k_undim_Cro": 0.055,  # /min
    "delta_CI2": 0.010,    # /min
    "delta_Cro2": 0.012,   # /min

    # RecA-driven CI cleavage (SOS induction), weak under basal conditions
    # Basal cleavage rate chosen to limit CI fragment accumulation in the lysogenic regime
    "k_cleave": 2.0e-7,   # 1/(molecule*min)
    "delta_CIc": 0.012,   # /min (clearance of cleaved fragments)

    # RecA dynamics with a low basal setpoint representing the no-damage regime
    "RecA0": 120.0,
    "k_RecA_relax": 0.0010,  # /min
}

# Initial condition biased toward lysogenic state (CI-high, Cro-low).
Y0 = [
    55.0,     # mCI
    6.0,      # mCro
    2600.0,   # CI
    60.0,     # Cro
    420.0,    # CI2
    8.0,      # Cro2
    0.0,      # CIc
    120.0,    # RecA
]

TSPAN = (0.0, 3000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    mCI, mCro, CI, Cro, CI2, Cro2, CIc, RecA = Y

    # Dimer-mediated repression
    rep_Cro = 1.0 / (1.0 + (CI2 / p["K_CI_rep_Cro"]) ** p["n_rep"])   # CI2 represses cro
    rep_CI = 1.0 / (1.0 + (Cro2 / p["K_Cro_rep_CI"]) ** p["n_rep"])   # Cro2 represses cI

    # CI positive autoregulation
    act_CI = (CI2 ** p["n_CI_act"]) / (p["K_CI_act"] ** p["n_CI_act"] + CI2 ** p["n_CI_act"])
    act_factor = 1.0 + p["act_gain"] * act_CI

    v_tx_CI = p["alpha_leak"] + (p["alpha_CI_max"] * rep_CI) * act_factor
    v_tx_Cro = p["alpha_leak"] + (p["alpha_Cro_max"] * rep_Cro)

    dmCI = v_tx_CI - p["delta_m"] * mCI
    dmCro = v_tx_Cro - p["delta_m"] * mCro

    # Translation
    v_tl_CI = p["k_tl_CI"] * mCI
    v_tl_Cro = p["k_tl_Cro"] * mCro

    # Dimerization / undimerization
    v_dim_CI = p["k_dim_CI"] * CI * CI
    v_undim_CI = p["k_undim_CI"] * CI2

    v_dim_Cro = p["k_dim_Cro"] * Cro * Cro
    v_undim_Cro = p["k_undim_Cro"] * Cro2

    # RecA-mediated cleavage of CI dimer
    v_cleave = p["k_cleave"] * RecA * CI2

    dCI = v_tl_CI - 2.0 * v_dim_CI + 2.0 * v_undim_CI - p["delta_CI"] * CI
    dCro = v_tl_Cro - 2.0 * v_dim_Cro + 2.0 * v_undim_Cro - p["delta_Cro"] * Cro

    dCI2 = v_dim_CI - v_undim_CI - p["delta_CI2"] * CI2 - v_cleave
    dCro2 = v_dim_Cro - v_undim_Cro - p["delta_Cro2"] * Cro2

    dCIc = v_cleave - p["delta_CIc"] * CIc

    # RecA: slow relaxation to basal
    dRecA = p["k_RecA_relax"] * (p["RecA0"] - RecA)

    return np.array([dmCI, dmCro, dCI, dCro, dCI2, dCro2, dCIc, dRecA], dtype=float)

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
    plt.title("grn07_lambda_switch (lysogenic default; low basal RecA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn07_lambda_switch_preview.png", dpi=220)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
