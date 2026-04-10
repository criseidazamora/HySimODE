# ============================================================================
# Family 5 (Gene Regulatory Circuits) — Lambda switch with higher-order cooperativity
# grn08_lambda_switch_cooperative.py
# CI–Cro mutual repression with CI positive autoregulation and explicit
# dimerization (CI2, Cro2).
# RecA drives CI2 cleavage into CIc.
# Higher-order cooperativity is represented through elevated Hill exponents.
# The default basal regime is lysogenic (CI-high, Cro-low), and a smooth RecA
# pulse can induce switching-like transients.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "grn08_lambda_switch_cooperative"

SPECIES_NAMES = [
    "mCI",    # 0
    "mCro",   # 1
    "CI",     # 2
    "Cro",    # 3
    "CI2",    # 4
    "Cro2",   # 5
    "CIc",    # 6
    "RecA",   # 7
]

PARAMS = {
    # Transcription (molecules/min)
    "alpha_CI_max": 13.0,
    "alpha_Cro_max": 16.0,
    "alpha_leak": 0.05,

    # CI autoregulation (activation by CI2)
    "K_CI_act": 520.0,
    "n_CI_act": 3.0,
    "act_gain": 1.0,

    # Mutual repression thresholds and cooperativity (dimers)
    "K_CI_rep_Cro": 180.0,  # CI2 represses cro
    "K_Cro_rep_CI": 220.0,  # Cro2 represses cI
    "n_rep": 4.5,

    # mRNA degradation
    "delta_m": 0.22,

    # Translation and monomer degradation
    "k_tl_CI": 5.0,
    "k_tl_Cro": 5.6,
    "delta_CI": 0.010,
    "delta_Cro": 0.012,

    # Dimerization / undimerization
    "k_dim_CI": 2.2e-6,
    "k_undim_CI": 0.060,
    "k_dim_Cro": 6.0e-6,
    "k_undim_Cro": 0.055,
    "delta_CI2": 0.010,
    "delta_Cro2": 0.012,

    # RecA-driven CI2 cleavage
    "k_cleave": 1.0e-5,
    "delta_CIc": 0.012,

    # RecA dynamics
    "RecA0": 120.0,
    "k_RecA_relax": 0.0010,

    # Smooth pulse (minutes)
    "pulse_on": 600.0,
    "pulse_off": 1100.0,
    "pulse_amp": 12000.0,
    "pulse_tau": 20.0,
}

# Lysogenic-biased initial condition
Y0 = np.array([
    55.0,     # mCI
    6.0,      # mCro
    2600.0,   # CI
    60.0,     # Cro
    420.0,    # CI2
    8.0,      # Cro2
    0.0,      # CIc
    120.0,    # RecA
], dtype=float)

TSPAN = (0.0, 3000.0)

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def recA_target(t: float, p: dict) -> float:
    """Basal RecA0 + smooth rectangular pulse."""
    ton = p["pulse_on"]
    toff = p["pulse_off"]
    tau = p["pulse_tau"]
    amp = p["pulse_amp"]
    up = sigmoid((t - ton) / tau)
    down = sigmoid((t - toff) / tau)
    return p["RecA0"] + amp * (up - down)

def dYdt_core(t: float, Y: np.ndarray, use_pulse: bool) -> np.ndarray:
    p = PARAMS
    mCI, mCro, CI, Cro, CI2, Cro2, CIc, RecA = Y

    # Dimer-mediated repression (higher cooperativity)
    rep_Cro = 1.0 / (1.0 + (CI2 / p["K_CI_rep_Cro"]) ** p["n_rep"])
    rep_CI  = 1.0 / (1.0 + (Cro2 / p["K_Cro_rep_CI"]) ** p["n_rep"])

    # CI positive autoregulation
    act_CI = (CI2 ** p["n_CI_act"]) / (p["K_CI_act"] ** p["n_CI_act"] + CI2 ** p["n_CI_act"])
    act_factor = 1.0 + p["act_gain"] * act_CI

    v_tx_CI  = p["alpha_leak"] + (p["alpha_CI_max"] * rep_CI) * act_factor
    v_tx_Cro = p["alpha_leak"] + (p["alpha_Cro_max"] * rep_Cro)

    dmCI  = v_tx_CI  - p["delta_m"] * mCI
    dmCro = v_tx_Cro - p["delta_m"] * mCro

    # Translation
    v_tl_CI  = p["k_tl_CI"]  * mCI
    v_tl_Cro = p["k_tl_Cro"] * mCro

    # Dimerization / undimerization
    v_dim_CI    = p["k_dim_CI"] * CI * CI
    v_undim_CI  = p["k_undim_CI"] * CI2
    v_dim_Cro   = p["k_dim_Cro"] * Cro * Cro
    v_undim_Cro = p["k_undim_Cro"] * Cro2

    # RecA-mediated cleavage of CI2
    v_cleave = p["k_cleave"] * RecA * CI2

    dCI   = v_tl_CI  - 2.0*v_dim_CI  + 2.0*v_undim_CI  - p["delta_CI"]  * CI
    dCro  = v_tl_Cro - 2.0*v_dim_Cro + 2.0*v_undim_Cro - p["delta_Cro"] * Cro
    dCI2  = v_dim_CI  - v_undim_CI  - p["delta_CI2"]  * CI2  - v_cleave
    dCro2 = v_dim_Cro - v_undim_Cro - p["delta_Cro2"] * Cro2

    dCIc = v_cleave - p["delta_CIc"] * CIc

    # RecA: relax to target
    target = recA_target(t, p) if use_pulse else p["RecA0"]
    dRecA = p["k_RecA_relax"] * (target - RecA)

    return np.array([dmCI, dmCro, dCI, dCro, dCI2, dCro2, dCIc, dRecA], dtype=float)

# ---------------------------
# Model interface used by the RFC pipeline
# ---------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    """
    Pipeline-exposed ODE system corresponding to the pulsed RecA regime.
    """
    return dYdt_core(t, y, use_pulse=True)

# Coupled basal and pulsed simulation in a single solver call
def odes_stacked_one_solver(t, y16):
    """
    Integrate basal and pulsed regimes in one stacked solve_ivp system:
    y16 = [Y_basal(8), Y_pulsed(8)]
    """
    Yb = y16[:8]
    Yp = y16[8:]
    dYb = dYdt_core(t, Yb, use_pulse=False)
    dYp = dYdt_core(t, Yp, use_pulse=True)
    return np.concatenate([dYb, dYp])

if __name__ == "__main__":
    t_eval = np.linspace(TSPAN[0], TSPAN[1], 2000)

    # Single solver call: basal + pulsed together
    y0_16 = np.concatenate([Y0, Y0])
    sol = solve_ivp(
        odes_stacked_one_solver,
        TSPAN,
        y0_16,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-7,
        atol=1e-9,
    )

    t = sol.t
    Yb = sol.y[:8, :]   # basal
    Yp = sol.y[8:, :]   # pulsed

    # Micro-audit computed on pulsed trajectories
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        y = Yp[i]
        q80 = np.quantile(y, 0.80)
        q99 = np.quantile(y, 0.99)
        label = int((q80 < 200) and (q99 < 200))
        labels.append(label)
        print(f"{name}: q80={q80:.2f}, q99={q99:.2f}, label={label}")
    print(f"Summary: label1={sum(labels)}, label0={len(labels)-sum(labels)}")
    if sum(labels) == 0 or sum(labels) == len(labels):
        print("WARNING: monoclass model (may not be useful for training)")

    # Plot basal
    for i in range(8):
        plt.plot(t, Yb[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("grn08_lambda_switch_cooperative (lysogenic default expected)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn08_lambda_switch_cooperative_basal.png", dpi=220)
    plt.close()

    # Plot pulsed
    for i in range(8):
        plt.plot(t, Yp[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("grn08_lambda_switch_cooperative (RecA pulse perturbation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn08_lambda_switch_cooperative_pulse.png", dpi=220)
    plt.close()

    # Pulse effect metrics
    print("\n=== Pulse effect metrics (pulsed - basal) ===")
    for i, name in enumerate(SPECIES_NAMES):
        diff = Yp[i] - Yb[i]
        delta_final = diff[-1]
        delta_max = np.max(diff)
        delta_min = np.min(diff)
        auc_diff = np.trapezoid(diff, t)  
        denom = max(1e-12, abs(Yb[i, -1]))
        rel_final = delta_final / denom
        print(
            f"{name:>4s} | Δfinal={delta_final: .2f}  "
            f"relΔfinal={rel_final: .3f}  "
            f"Δmax={delta_max: .2f}  Δmin={delta_min: .2f}  "
            f"AUCΔ={auc_diff: .2e}"
        )

    # Switch detector: Cro2/CI2 ratio crossing threshold
    iCI2 = SPECIES_NAMES.index("CI2")
    iCro2 = SPECIES_NAMES.index("Cro2")
    ratio_p = (Yp[iCro2] + 1e-9) / (Yp[iCI2] + 1e-9)
    thresh = 1.0
    idx = np.where(ratio_p > thresh)[0]
    if len(idx) > 0:
        print(f"\nSwitch detected: Cro2/CI2 > {thresh} at t = {t[idx[0]]:.1f} min")
    else:
        print(f"\nNo switch detected: Cro2/CI2 never exceeded {thresh}")

    # Delta plot for key species
    key = ["CI", "Cro", "CI2", "Cro2", "RecA"]
    for k in key:
        i = SPECIES_NAMES.index(k)
        plt.plot(t, Yp[i] - Yb[i], label=f"Δ{k} (pulse-basal)")
    plt.axvline(PARAMS["pulse_on"], linestyle="--")
    plt.axvline(PARAMS["pulse_off"], linestyle="--")
    plt.xlabel("time [min]")
    plt.ylabel("Δ molecules")
    plt.title("Pulse effect (pulsed - basal) for key species")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn08_lambda_cooperative_pulse_effect.png", dpi=220)
    plt.close()

# ------------------------------------------------------------
# Pipeline interface
# ------------------------------------------------------------

def dYdt(t, Y):
    return dYdt_core(t, Y, use_pulse=True)

# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)

