# osc04_goodwin_delay_chain.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 4 (Deterministic Oscillators) — Goodwin oscillator with explicit delay chain
# CORRECTED: 5-step distributed delay, strengthened negative feedback, tuned for sustained oscillations
#            (high Hill n, strong fold-change, linear degradation only).
# ============================================================================
MODEL_NAME = "osc04_goodwin_delay_chain"

SPECIES_NAMES = [
    "mRNA",  # S0: transcript (low copy)
    "P0",    # S1: translated protein (high copy)
    "P1",    # S2: delay intermediate 1 (high copy)
    "P2",    # S3: delay intermediate 2 (high copy)
    "P3",    # S4: delay intermediate 3 (mid copy)
    "P4",    # S5: delay intermediate 4 (mid copy)
    "Rep",   # S6: active repressor (low-to-mid copy)
]

# ===== PARAMETERS ============================================================
# Canonical Goodwin with distributed delay (5 intermediates):
#   dm/dt  = alpha_leak + alpha_max/(1 + (Rep/K)^n) - delta_m*m
#   chain: m -> P0 -> P1 -> P2 -> P3 -> P4 -> Rep (all linear), all linear degradation
#
# Tuned to produce visible oscillations in ~500–2000 min (typically sustained / weakly damped to a limit cycle).
PARAMS = {
    # Strong transcriptional repression (gain)
    "alpha_max": 200.0,   # molecules/min
    "alpha_leak": 0.30,   # molecules/min (strong fold-change)
    "K_rep": 850.0,       # molecules
    "n_rep": 12.0,        # cooperativity (>= 8; strengthened for oscillations)

    # Linear degradation (sets timescales)
    "delta_m": 0.028,     # /min
    "delta_p": 0.0015,    # /min
    "delta_rep": 0.0016,  # /min

    # Translation and delay chain (distributed delay: slower conversions)
    "k_tl": 0.60,         # molecules/min per mRNA
    "k01": 0.0035,        # /min
    "k12": 0.0035,        # /min
    "k23": 0.0035,        # /min
    "k34": 0.0035,        # /min
    "k45": 0.0035,        # /min
}

# ===== INITIAL CONDITIONS ====================================================
# Moderate ICs (avoid one-shot giant transient dominating the run)
Y0 = [
    85.0,     # mRNA
    7000.0,   # P0
    4200.0,   # P1
    2600.0,   # P2
    1900.0,   # P3
    1300.0,   # P4
    950.0,    # Rep
]

TSPAN = (0.0, 6000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    m, P0, P1, P2, P3, P4, Rep = Y

    # High-n Hill repression (Goodwin core)
    rep_term = 1.0 / (1.0 + (Rep / p["K_rep"]) ** p["n_rep"])
    v_tx = p["alpha_leak"] + p["alpha_max"] * rep_term
    v_mdeg = p["delta_m"] * m

    # Translation + distributed delay chain
    v_tl = p["k_tl"] * m
    v01 = p["k01"] * P0
    v12 = p["k12"] * P1
    v23 = p["k23"] * P2
    v34 = p["k34"] * P3
    v45 = p["k45"] * P4

    # Linear degradation
    v_P0_deg = p["delta_p"] * P0
    v_P1_deg = p["delta_p"] * P1
    v_P2_deg = p["delta_p"] * P2
    v_P3_deg = p["delta_p"] * P3
    v_P4_deg = p["delta_p"] * P4
    v_Rep_deg = p["delta_rep"] * Rep

    dm = v_tx - v_mdeg
    dP0 = v_tl - v01 - v_P0_deg
    dP1 = v01 - v12 - v_P1_deg
    dP2 = v12 - v23 - v_P2_deg
    dP3 = v23 - v34 - v_P3_deg
    dP4 = v34 - v45 - v_P4_deg
    dRep = v45 - v_Rep_deg

    return np.array([dm, dP0, dP1, dP2, dP3, dP4, dRep], dtype=float)

# ===== SELF-TEST / MICRO-AUDIT ==============================================
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

    # Plot (required)
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("osc04_goodwin_delay_chain (Goodwin with 5-step distributed delay; n=12; linear decay)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("osc04_goodwin_delay_chain_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
