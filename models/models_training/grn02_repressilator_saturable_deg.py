# grn02_repressilator_saturable_deg.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 5 (Gene Regulatory Circuits) — Repressilator with saturable protein degradation
# Three genes in a ring: A represses B, B represses C, C represses A
# Protein degradation is Michaelis–Menten (capacity-limited proteolysis).
# ============================================================================
MODEL_NAME = "grn02_repressilator_saturable_deg"

SPECIES_NAMES = [
    "mA", "mB", "mC",   # mRNAs (low copy)
    "pA", "pB", "pC",   # proteins (higher copy; degraded saturably)
]

PARAMS = {
    # Transcription (Hill repression by upstream protein)
    "alpha_max": 20.0,   # molecules/min
    "alpha_leak": 0.25,  # molecules/min
    "K_rep": 600.0,      # molecules
    "n_rep": 3.0,        # cooperativity

    # mRNA degradation (linear)
    "delta_m": 0.20,     # /min

    # Translation
    "k_tl": 6.5,         # molecules/min per mRNA

    # Basal protein dilution (linear, weak)
    "delta_p": 0.0035,   # /min

    # Saturable protein degradation (Michaelis–Menten form)
    # v_deg(pX) = Vmax * pX/(Km + pX)
    "Vmax_deg": 95.0,    # molecules/min (protease capacity)
    "Km_deg": 1400.0,    # molecules
}

Y0 = [
    40.0,  12.0,  55.0,   # mA, mB, mC
    1200.0, 2800.0, 800.0 # pA, pB, pC
]

TSPAN = (0.0, 5000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    mA, mB, mC, pA, pB, pC = Y

    # Ring repression
    repA = 1.0 / (1.0 + (pC / p["K_rep"]) ** p["n_rep"])  # C represses A
    repB = 1.0 / (1.0 + (pA / p["K_rep"]) ** p["n_rep"])  # A represses B
    repC = 1.0 / (1.0 + (pB / p["K_rep"]) ** p["n_rep"])  # B represses C

    v_tx_A = p["alpha_leak"] + p["alpha_max"] * repA
    v_tx_B = p["alpha_leak"] + p["alpha_max"] * repB
    v_tx_C = p["alpha_leak"] + p["alpha_max"] * repC

    dmA = v_tx_A - p["delta_m"] * mA
    dmB = v_tx_B - p["delta_m"] * mB
    dmC = v_tx_C - p["delta_m"] * mC

    # Translation
    v_tl_A = p["k_tl"] * mA
    v_tl_B = p["k_tl"] * mB
    v_tl_C = p["k_tl"] * mC

    # Saturable proteolysis + weak dilution
    v_deg_A = p["Vmax_deg"] * (pA / (p["Km_deg"] + pA))
    v_deg_B = p["Vmax_deg"] * (pB / (p["Km_deg"] + pB))
    v_deg_C = p["Vmax_deg"] * (pC / (p["Km_deg"] + pC))

    dpA = v_tl_A - v_deg_A - p["delta_p"] * pA
    dpB = v_tl_B - v_deg_B - p["delta_p"] * pB
    dpC = v_tl_C - v_deg_C - p["delta_p"] * pC

    return np.array([dmA, dmB, dmC, dpA, dpB, dpC], dtype=float)

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

    # Optional preview plot (sanity check)
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("grn02_repressilator_saturable_deg (repressilator with MM protein degradation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn02_repressilator_saturable_deg_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
