# grn06_toggle_asymmetric.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 5 — Gene Regulatory Circuits (Low Copy / Noise-Prone)
# Model 26: Asymmetric genetic toggle switch WITH bistability
# - Mutual repression (Hill) with strong cooperativity and low leak.
# - Mild asymmetry (alpha_mA > alpha_mB) biases, but does not eliminate, the B-high state.
# - Inducers act as tunable inactivation; here kept moderate to preserve bistability.
# ============================================================================
MODEL_NAME = "grn06_toggle_asymmetric"

SPECIES_NAMES = [
    "mA",    # S0: mRNA for repressor A (low copy)
    "mB",    # S1: mRNA for repressor B (low copy)
    "pA",    # S2: protein A (repressor)
    "pB",    # S3: protein B (repressor)
    "IndA",  # S4: inducer that inactivates A (high copy-like input)
    "IndB",  # S5: inducer that inactivates B (high copy-like input)
]

PARAMS = {
    # Transcription (molecules/min)
    # Mild asymmetry: A slightly stronger than B, but both can dominate depending on initial state.
    "alpha_mA": 6.2,
    "alpha_mB": 5.1,
    "alpha_leak": 0.01,

    # Repression thresholds and cooperativity (Hill repression)
    # Symmetric-ish repression is key for bistability; slight asymmetry kept via K values and alphas.
    "K_A": 2500.0,   # pB inhibits A
    "K_B": 2200.0,   # pA inhibits B
    "n_A": 4.2,
    "n_B": 4.2,

    # mRNA degradation (/min)
    "delta_m": 0.25,

    # Translation and protein degradation
    "beta_pA": 9.0,   # molecules/min per mRNA
    "beta_pB": 9.0,
    "delta_pA": 0.010,
    "delta_pB": 0.010,

    # Inducer modulation: effective free repressor decreases with inducer
    # Keep moderate so the toggle remains nonlinear and bistable.
    "K_I_A": 4500.0,   # IndA half-inactivates pA
    "K_I_B": 4500.0,   # IndB half-inactivates pB

    # Inducer dynamics (slow relaxation to setpoints)
    # Moderate basal inputs (not huge), preserving repression strength.
    "IndA0": 1400.0,
    "IndB0": 1400.0,
    "k_ind_relax": 0.0010,  # /min
}

# Default initial condition (A-leaning but not extreme)
Y0 = [
    50.0,    # mA
    50.0,    # mB
    4000.0,  # pA
    4000.0,  # pB
    1400.0,  # IndA
    1400.0,  # IndB
]

TSPAN = (0.0, 6000.0)  # minutes

def hill_rep(x, K, n):
    return 1.0 / (1.0 + (x / K) ** n)

def dYdt(t, Y):
    p = PARAMS
    mA, mB, pA, pB, IndA, IndB = Y

    # Inducers reduce effective (DNA-binding) free repressors
    pA_eff = pA / (1.0 + (IndA / p["K_I_A"]))
    pB_eff = pB / (1.0 + (IndB / p["K_I_B"]))

    # Mutual repression
    txA = p["alpha_leak"] + p["alpha_mA"] * hill_rep(pB_eff, p["K_A"], p["n_A"])
    txB = p["alpha_leak"] + p["alpha_mB"] * hill_rep(pA_eff, p["K_B"], p["n_B"])

    dmA = txA - p["delta_m"] * mA
    dmB = txB - p["delta_m"] * mB

    dpA = p["beta_pA"] * mA - p["delta_pA"] * pA
    dpB = p["beta_pB"] * mB - p["delta_pB"] * pB

    # Inducers: slow relaxation to setpoints (environmental control)
    dIndA = p["k_ind_relax"] * (p["IndA0"] - IndA)
    dIndB = p["k_ind_relax"] * (p["IndB0"] - IndB)

    return np.array([dmA, dmB, dpA, dpB, dIndA, dIndB], dtype=float)

if __name__ == "__main__":
    t_span = TSPAN
    t_eval = np.linspace(t_span[0], t_span[1], 2000)

    # Primary deterministic solve (used for micro-audit)
    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    # Micro-audit: per-species q80/q99 labels (fixed rule)
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

    # Plot primary trajectory
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("grn06_toggle_asymmetric (bistable toggle; default trajectory)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn06_toggle_asymmetric_preview.png", dpi=220)
    plt.close()

    # --------------------------------------------------------------------
    # Bistability comparison (two different initial conditions)
    # This is for visual confirmation of A-high vs B-high attractors.
    # --------------------------------------------------------------------
    Y0_Ahigh = [
        65.0,    # mA
        25.0,    # mB
        5200.0,  # pA
        900.0,   # pB
        PARAMS["IndA0"],
        PARAMS["IndB0"],
    ]
    Y0_Bhigh = [
        25.0,    # mA
        65.0,    # mB
        900.0,   # pA
        5200.0,  # pB
        PARAMS["IndA0"],
        PARAMS["IndB0"],
    ]

    solA = solve_ivp(dYdt, t_span, Y0_Ahigh, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)
    solB = solve_ivp(dYdt, t_span, Y0_Bhigh, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    plt.plot(solA.t, solA.y[2], label="pA (A-high init)")
    plt.plot(solA.t, solA.y[3], label="pB (A-high init)")
    plt.plot(solB.t, solB.y[2], label="pA (B-high init)")
    plt.plot(solB.t, solB.y[3], label="pB (B-high init)")
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("grn06_toggle_asymmetric — bistability check (two basins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn06_toggle_asymmetric_bistability.png", dpi=220)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py / feature scripts
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
