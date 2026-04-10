# ============================================================================
# Family 6 (Motif-Hybrid / Multiscale) — Delayed negative-feedback pulse motif
# motif06_delayed_negative_feedback_pulse.py
# Activator–inhibitor circuit with positive autoregulation, delayed negative
# feedback, and sequestration-mediated buffering. A nutrient-driven product
# module introduces a multiscale high-copy background coupled to the regulatory
# core.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "motif06_delayed_negative_feedback_pulse"

SPECIES_NAMES = [
    "mA",        # S0: activator mRNA (low copy)
    "A",         # S1: activator protein (moderate copy, oscillatory)
    "mI",        # S2: inhibitor mRNA (low copy)
    "I",         # S3: inhibitor protein (moderate copy)
    "D_delay",   # S4: delayed inhibitory effector derived from I (introduces lag)
    "C_AD",      # S5: sequestration complex A:D (buffering / ultrasensitivity)
    "N_pool",    # S6: nutrient / energy pool (high copy chemostat-like driver)
    "P_product"  # S7: metabolic product (can accumulate to high copy)
]

PARAMS = {
    # Activator transcription: positive autoregulation + delayed repression by D_delay
    "basal_A": 0.25,      # molecules/time
    "alpha_A": 22.0,      # molecules/time (max induced transcription)
    "K_Aact": 35.0,       # molecules
    "n_Aact": 3.0,        # Hill coefficient (positive feedback)

    "K_Drep": 25.0,       # molecules
    "n_Drep": 3.0,        # Hill coefficient (repression steepness)

    # Inhibitor transcription activated by A (feed-forward to negative feedback arm)
    "basal_I": 0.15,      # molecules/time
    "alpha_I": 18.0,      # molecules/time
    "K_Iact": 25.0,       # molecules
    "n_Iact": 2.0,        # Hill coefficient

    # mRNA turnover (fast)
    "delta_mA": 1.10,     # /time
    "delta_mI": 1.00,     # /time

    # Translation
    "k_tlA": 7.0,         # molecules/time per mRNA
    "k_tlI": 6.0,         # molecules/time per mRNA

    # Protein degradation (slower than mRNA)
    "delta_A": 0.12,      # /time
    "delta_I": 0.10,      # /time

    # Delay module: I -> D_delay (slow maturation / modification)
    "k_delay": 0.22,      # /time
    "delta_D": 0.06,      # /time

    # Sequestration (A + D <-> C_AD) adds buffering + nonlinearity
    "k_bind": 0.010,      # 1/(molecule*time)
    "k_unbind": 0.18,     # /time
    "delta_C": 0.04,      # /time

    # Nutrient/product module (multiscale, high-copy background)
    "k_inN": 4.0,         # molecules/time (feed)
    "delta_N": 0.010,     # /time (slow leak)
    "k_cons": 0.020,      # 1/(molecule*time) effective consumption coupling
    "K_N": 120.0,         # molecules (Michaelis-like saturation)

    "k_prod": 1.10,       # molecules/time scaling (product formation)
    "delta_P": 0.020      # /time
}

# Initial conditions: low-copy transcripts, moderate A/I, high nutrient pool
Y0 = [
    2.0,     # mA
    18.0,    # A
    1.5,     # mI
    10.0,    # I
    6.0,     # D_delay
    0.0,     # C_AD
    650.0,   # N_pool (high copy -> likely label=0)
    20.0     # P_product
]

TSPAN = (0.0, 300.0)


def dYdt(t, Y):
    p = PARAMS
    mA, A, mI, I, D, C, N, P = Y

    # --- Regulatory nonlinearities ---
    # Positive autoregulation of A
    actA = (A ** p["n_Aact"]) / (p["K_Aact"] ** p["n_Aact"] + A ** p["n_Aact"])

    # Delayed repression by D (steep Hill)
    repD = 1.0 / (1.0 + (D / p["K_Drep"]) ** p["n_Drep"])

    # A activates I transcription
    actI = (A ** p["n_Iact"]) / (p["K_Iact"] ** p["n_Iact"] + A ** p["n_Iact"])

    # --- Transcription ---
    v_tx_A = p["basal_A"] + p["alpha_A"] * actA * repD
    v_tx_I = p["basal_I"] + p["alpha_I"] * actI

    dmA = v_tx_A - p["delta_mA"] * mA
    dmI = v_tx_I - p["delta_mI"] * mI

    # --- Translation ---
    v_tlA = p["k_tlA"] * mA
    v_tlI = p["k_tlI"] * mI

    # --- Sequestration fluxes (buffering / ultrasensitivity) ---
    v_bind = p["k_bind"] * A * D
    v_unbind = p["k_unbind"] * C

    # --- Proteins / delay species ---
    dA = v_tlA - p["delta_A"] * A - v_bind + v_unbind
    dI = v_tlI - p["delta_I"] * I

    # Delay: I -> D (maturation) and sequestration with A
    dD = p["k_delay"] * I - p["delta_D"] * D - v_bind + v_unbind

    dC = v_bind - v_unbind - p["delta_C"] * C

    # --- Nutrient/product module (multiscale driver) ---
    # Saturable coupling of nutrient usage to A activity
    satN = N / (p["K_N"] + N)
    v_cons = p["k_cons"] * A * satN
    dN = p["k_inN"] - v_cons - p["delta_N"] * N

    # Product accumulates from A-dependent flux, then decays slowly
    v_prod = p["k_prod"] * v_cons
    dP = v_prod - p["delta_P"] * P

    return np.array([dmA, dA, dmI, dI, dD, dC, dN, dP], dtype=float)


if __name__ == "__main__":
    t_span = TSPAN
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau")

    # ----------------------------
    # Micro-audit: q80/q99 labels
    # ----------------------------
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        y = sol.y[i]
        q80 = np.quantile(y, 0.80)
        q99 = np.quantile(y, 0.99)
        label = int((q80 < 200) and (q99 < 200))
        labels.append(label)
        print(f"{name}: q80={q80:.3f}, q99={q99:.3f} => label={label}")

    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass model")

    # ----------------------------
    # Plot: all trajectories
    # ----------------------------
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("motif06_delayed_negative_feedback_pulse")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_preview.png", dpi=220)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
