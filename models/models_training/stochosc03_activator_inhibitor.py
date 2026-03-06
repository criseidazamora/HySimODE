import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# Family 7 — Noise-Prone Oscillatory Models
# Model 36: stochosc03_activator_inhibitor.py (DETERMINISTIC ONLY)
#
# Activator–inhibitor oscillator with explicit positive feedback + delayed inhibition.
# Structure: mA, A, mI, I, D_delay, C_AD, N_pool, P_product
# =============================================================================

MODEL_NAME = "stochosc03_activator_inhibitor_deterministic"

SPECIES_NAMES = [
    "mA",       # 0
    "A",        # 1
    "mI",       # 2
    "I",        # 3
    "D_delay",  # 4
    "C_AD",     # 5
    "N_pool",   # 6
    "P_product" # 7
]

PARAMS = {
    # A transcription: leak + auto-activation * repression-by-D * nutrient availability
    "alphaA_leak": 0.05,     # molecules/min
    "alphaA_max":  14.0,     # molecules/min
    "K_Aact":      55.0,     # molecules
    "n_Aact":      3.0,      # cooperativity

    "K_Drep":      90.0,     # molecules
    "n_Drep":      4.0,      # ultrasensitive repression

    # Nutrient scaling of transcription
    "K_N_tx":      200.0,    # molecules

    # I transcription: induced by A
    "alphaI_leak": 0.02,
    "alphaI_max":  9.0,
    "K_Iact":      85.0,
    "n_Iact":      3.0,

    # mRNA decay (fast)
    "delta_mA":    0.60,     # /min
    "delta_mI":    0.60,     # /min

    # Translation
    "k_tlA":       7.0,      # molecules/min per mRNA
    "k_tlI":       6.0,      # molecules/min per mRNA

    # Protein / effector turnover
    "delta_A":     0.10,     # /min
    "delta_I":     0.12,     # /min
    "delta_D":     0.025,    # /min (slow delay arm)

    # Delay: maturation I -> D_delay
    "k_mat":       0.035,    # /min

    # Sequestration A + D <-> C_AD
    "k_on":        2.0e-4,   # 1/(molecule*min)
    "k_off":       0.25,     # /min
    "delta_C":     0.12,     # /min

    # Metabolic context: homeostatic N_pool + A-dependent consumption
    "N0":          600.0,    # molecules
    "k_in":        0.010,    # /min (relaxation to N0)

    "k_cons":      3.0,      # molecules/min
    "K_cons":      90.0,     # molecules
    "n_cons":      2.0,
    "K_N_cons":    200.0,    # molecules

    # Product
    "k_prod":      0.8,
    "delta_P":     0.015,    # /min
}

Y0 = np.array([
    0.8,    # mA
    30.0,   # A
    0.3,    # mI
    10.0,   # I
    40.0,   # D_delay
    0.0,    # C_AD
    600.0,  # N_pool
    0.0     # P_product
], dtype=float)

TSPAN = (0.0, 600.0)

def hill_act(x, K, n):
    x = max(x, 0.0)
    return (x**n) / (K**n + x**n)

def hill_rep(x, K, n):
    x = max(x, 0.0)
    return 1.0 / (1.0 + (x / K)**n)

def rhs(t, y, p=PARAMS):
    mA, A, mI, I, D, C, N, P = y

    # Transcription
    txA = (
        p["alphaA_leak"]
        + p["alphaA_max"]
        * hill_act(A, p["K_Aact"], p["n_Aact"])
        * hill_rep(D, p["K_Drep"], p["n_Drep"])
        * (N / (p["K_N_tx"] + N))
    )
    txI = p["alphaI_leak"] + p["alphaI_max"] * hill_act(A, p["K_Iact"], p["n_Iact"])

    dmA = txA - p["delta_mA"] * mA
    dmI = txI - p["delta_mI"] * mI

    # Translation
    prodA = p["k_tlA"] * mA
    prodI = p["k_tlI"] * mI

    # Sequestration A + D <-> C
    v_on  = p["k_on"] * A * D
    v_off = p["k_off"] * C

    dA = prodA - p["delta_A"] * A - v_on + v_off
    dI = prodI - p["delta_I"] * I
    dD = p["k_mat"] * I - p["delta_D"] * D - v_on + v_off
    dC = v_on - v_off - p["delta_C"] * C

    # Metabolic resource consumption (A-dependent, saturable in A and N)
    v_cons = p["k_cons"] * hill_act(A, p["K_cons"], p["n_cons"]) * (N / (p["K_N_cons"] + N))

    # Homeostatic replenishment to N0 (prevents monotonic drift)
    dN = p["k_in"] * (p["N0"] - N) - v_cons

    # Product
    dP = p["k_prod"] * v_cons - p["delta_P"] * P

    return np.array([dmA, dA, dmI, dI, dD, dC, dN, dP], dtype=float)

def simulate_deterministic():
    t_eval = np.linspace(TSPAN[0], TSPAN[1], 6000)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, PARAMS),
        TSPAN, Y0, t_eval=t_eval,
        method="Radau",
        rtol=1e-7, atol=1e-9
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.t, sol.y

def micro_audit(traj):
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        yy = traj[i]
        q80 = np.quantile(yy, 0.80)
        q99 = np.quantile(yy, 0.99)
        label = int((q80 < 200.0) and (q99 < 200.0))
        labels.append(label)
        print(f"{name}: q80={q80:.3f}, q99={q99:.3f} => label={label}")
    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass model")

def make_plots(t, y):
    # Plot 1: regulatory module
    idx_reg = [0, 1, 2, 3, 4, 5]
    plt.figure(figsize=(12, 4), dpi=170)
    for i in idx_reg:
        plt.plot(t, y[i], label=SPECIES_NAMES[i])
    plt.title("stochosc03_activator_inhibitor — regulatory module (deterministic)")
    plt.xlabel("t")
    plt.ylabel("abundance / concentration")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("stochosc03_activator_inhibitor_regulatory.png", dpi=220)

    # Plot 2: metabolic context
    idx_met = [6, 7]
    plt.figure(figsize=(12, 4), dpi=170)
    for i in idx_met:
        plt.plot(t, y[i], label=SPECIES_NAMES[i])
    plt.title("stochosc03_activator_inhibitor — metabolic context (deterministic)")
    plt.xlabel("t")
    plt.ylabel("abundance / concentration")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("stochosc03_activator_inhibitor_metabolic.png", dpi=220)

    # Optional: overview
    plt.figure(figsize=(10, 5), dpi=160)
    for i in range(y.shape[0]):
        plt.plot(t, y[i], label=SPECIES_NAMES[i])
    plt.xlabel("t")
    plt.ylabel("abundance / concentration")
    plt.title("stochosc03_activator_inhibitor — overview (deterministic)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("stochosc03_activator_inhibitor_overview.png", dpi=220)

if __name__ == "__main__":
    t, y = simulate_deterministic()
    micro_audit(y)
    make_plots(t, y)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return rhs(t, y, PARAMS)
