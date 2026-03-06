# grn05_toggle_autoregulation_v2_balanced.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 5 (Gene Regulatory Circuits) — Toggle switch with autoregulation (ML + plausibility tuned)
# Mutual repression (A ⟂ B, B ⟂ A) plus positive autoregulation of A via A-dimer.
#
# Inducers (IndA/IndB): DOSE-normalized Gaussian pulses (finite area), with decay and weak consumption.
# Tuned so Ind pulses are strong but biologically plausible (order 10^3–10^4), not 10^5–10^6.
# Circuit parameters tuned to avoid the unrealistic symmetric coexistence and preserve switch-like behavior,
# while keeping diverse quantile patterns for ML (q80 vs q99).
# ============================================================================

MODEL_NAME = "grn05_toggle_autoregulation_v2_balanced"

SPECIES_NAMES = [
    "mA",   # S0: mRNA-A (low copy)
    "mB",   # S1: mRNA-B (low copy)
    "pA",   # S2: protein A monomer
    "pB",   # S3: protein B monomer
    "dA",   # S4: A dimer (regulatory form)
    "dB",   # S5: B dimer (regulatory form)
    "IndA", # S6: inducer reducing A activity (dynamic)
    "IndB", # S7: inducer reducing B activity (dynamic)
]

PARAMS = {
    # Transcription: mutual repression (by dimers) + A positive autoregulation (by dA)
    "alphaA_max": 18.0,   # molecules/min
    "alphaB_max": 18.0,   # molecules/min
    "alpha_leak": 0.20,   # molecules/min

    # Repression thresholds (brought back to realistic toggle regime)
    "K_Arep": 220.0,      # molecules (dB represses A)
    "K_Brep": 220.0,      # molecules (dA represses B)
    "n_rep": 3.0,         # cooperativity

    # Positive autoregulation of A by its own dimer (strong enough to support bistability)
    "K_Aact": 520.0,      # molecules (lowered so dA in 50–200 range has effect)
    "n_act": 2.0,
    "act_gain": 0.55,

    # mRNA degradation
    "delta_m": 0.22,      # /min

    # Translation and protein degradation (moderate copy numbers; avoid 10^4–10^5)
    "k_tl": 0.85,         # molecules/min per mRNA
    "delta_p": 0.028,     # /min

    # Dimerization / undimerization + turnover (keeps dimers in tens–hundreds)
    "k_dim": 1.1e-6,      # 1/(molecule*min)
    "k_undim": 0.050,     # /min
    "delta_d": 0.028,     # /min

    # Inducer potency scale (so pulses modulate but do not annihilate binding)
    "K_ind": 6000.0,      # molecules

    # ---- Inducers: dose-normalized pulses + decay + weak consumption (plausible scale) ----
    "IndA_base": 15.0,    # basal residual level
    "IndB_base": 15.0,
    "d_Ind": 0.012,       # /min (clears tail so q80 can remain low)

    # Pulse timing and width (minutes)
    "t_pulse_A": 260.0,
    "t_pulse_B": 1180.0,
    "pulse_sigma": 65.0,  # somewhat wider than 45 to avoid needle-like spikes

    # Total delivered doses (molecules)
    # Chosen so peak Ind levels are typically O(10^3–10^4), not O(10^5)
    "doseA": 14000.0,
    "doseB": 9000.0,

    # Weak consumption proxy (couples inducer to the expressed proteins)
    "k_ind_consume": 1.5e-6,  # 1/(molecule*min)
}

# Slight initial bias toward A to select an attractor, while still allowing perturbation by pulses
Y0 = [
    60.0,   # mA
    55.0,   # mB
    420.0,  # pA
    360.0,  # pB
    55.0,   # dA
    45.0,   # dB
    15.0,   # IndA
    15.0,   # IndB
]

TSPAN = (0.0, 6000.0)  # minutes


def dYdt(t, Y):
    p = PARAMS
    mA, mB, pA, pB, dA, dB, IndA, IndB = Y

    # ---- Inducer pulses: normalized Gaussian influx with finite area (dose) ----
    sigma = p["pulse_sigma"]
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    influx_A = p["doseA"] * norm * np.exp(-0.5 * ((t - p["t_pulse_A"]) / sigma) ** 2)
    influx_B = p["doseB"] * norm * np.exp(-0.5 * ((t - p["t_pulse_B"]) / sigma) ** 2)

    # Inducers reduce effective DNA-binding activity (dimensionless multipliers in (0,1])
    aA = 1.0 / (1.0 + IndA / p["K_ind"])
    aB = 1.0 / (1.0 + IndB / p["K_ind"])

    # Dimer-mediated mutual repression
    repA = 1.0 / (1.0 + (aB * dB / p["K_Arep"]) ** p["n_rep"])  # dB represses A
    repB = 1.0 / (1.0 + (aA * dA / p["K_Brep"]) ** p["n_rep"])  # dA represses B

    # Positive autoregulation of A by its own dimer (scaled by IndA via aA)
    actA_num = (aA * dA) ** p["n_act"]
    actA_den = (p["K_Aact"] ** p["n_act"] + (aA * dA) ** p["n_act"] + 1e-12)
    actA = actA_num / actA_den
    act_factor = 1.0 + p["act_gain"] * actA

    v_tx_A = p["alpha_leak"] + (p["alphaA_max"] * repA) * act_factor
    v_tx_B = p["alpha_leak"] + (p["alphaB_max"] * repB)

    dmA = v_tx_A - p["delta_m"] * mA
    dmB = v_tx_B - p["delta_m"] * mB

    # Translation
    v_tl_A = p["k_tl"] * mA
    v_tl_B = p["k_tl"] * mB

    # Dimerization / undimerization
    v_dim_A = p["k_dim"] * pA * pA
    v_dim_B = p["k_dim"] * pB * pB
    v_undim_A = p["k_undim"] * dA
    v_undim_B = p["k_undim"] * dB

    # Monomers: 2 monomers <-> 1 dimer
    dpA = v_tl_A - 2.0 * v_dim_A + 2.0 * v_undim_A - p["delta_p"] * pA
    dpB = v_tl_B - 2.0 * v_dim_B + 2.0 * v_undim_B - p["delta_p"] * pB

    # Dimers
    ddA = v_dim_A - v_undim_A - p["delta_d"] * dA
    ddB = v_dim_B - v_undim_B - p["delta_d"] * dB

    # Inducers: decay to baseline + finite-dose influx - weak consumption by target proteins
    dIndA = -p["d_Ind"] * (IndA - p["IndA_base"]) + influx_A - p["k_ind_consume"] * IndA * pA
    dIndB = -p["d_Ind"] * (IndB - p["IndB_base"]) + influx_B - p["k_ind_consume"] * IndB * pB

    return np.array([dmA, dmB, dpA, dpB, ddA, ddB, dIndA, dIndB], dtype=float)


if __name__ == "__main__":
    t_span = TSPAN
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau")

    # Micro-audit: per-species q80/q99 labels
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        y = sol.y[i]
        q80 = float(np.quantile(y, 0.80))
        q99 = float(np.quantile(y, 0.99))
        label = int((q80 < 200.0) and (q99 < 200.0))
        labels.append(label)
        print(f"{name}: q80={q80:.2f}, q99={q99:.2f}, label={label}")

    n1 = int(sum(labels))
    n0 = int(len(labels) - n1)
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass model (may not be useful for training)")

    # Preview plot
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("grn05_toggle_autoregulation_v2_balanced (plausible ML-tuned pulses)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grn05_toggle_autoregulation_v2_preview.png", dpi=220)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
