# motif05_stochastic_enzyme_deterministic_substrate.py
# Family 6 — Network Motifs / Multiscale
# Hybrid model: low-copy (effectively stochastic) enzyme statistics + abundant deterministic substrate/product pool.
# "Stochastic enzyme" is approximated via mean/variance dynamics (moment equations) with bursty production.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------
# (a) Variables globales obligatorias
# -------------------------
MODEL_NAME = "motif05_stochastic_enzyme_deterministic_substrate"

SPECIES_NAMES = [
    "E_mRNA",        # 0 fast, low copy (enzyme transcript)
    "E_mean",        # 1 low-copy enzyme mean abundance
    "E_var",         # 2 enzyme variance (noise proxy; moment-closure style)
    "I_inhibitor",   # 3 low/moderate inhibitor
    "C_EI",          # 4 enzyme-inhibitor complex (sequestration)
    "S_substrate",   # 5 abundant deterministic substrate pool
    "P_product",     # 6 product pool
    "R_regulator"    # 7 slow regulator that modulates E transcription (multiscale driver)
]

PARAMS = {
    # Regulator R (slow, low/moderate copy)
    "k_R_prod": 0.9,
    "d_R": 0.05,

    # E mRNA transcription controlled by R (Hill activation)
    "k_tx_E_max": 18.0,
    "K_RE": 18.0,
    "n_RE": 2.0,
    "d_mE": 3.2,

    # Enzyme mean dynamics (translation + dilution)
    "k_tl_E": 10.0,     # mean translation rate per mRNA
    "d_E": 0.55,        # enzyme dilution/degradation

    # Enzyme noise (variance) dynamics: bursty production approximated by Fano factor F_prod
    "F_prod": 6.0,      # >1 implies bursty translation; higher = noisier enzyme
    # For linear birth-death with input u and death d:
    # dVar/dt ≈ F*u + d*E_mean - 2*d*Var  (keeps Var ~ O(E_mean), not exploding)

    # Enzyme sequestration by inhibitor: E + I <-> C
    "k_on": 0.015,
    "k_off": 0.25,
    "d_C": 0.12,        # slow clearance of complex

    # Inhibitor homeostasis (moderate, slower than mRNA)
    "I_prod": 1.2,
    "d_I": 0.08,

    # Deterministic substrate/product turnover (abundant pool)
    "S_in": 10.0,
    "d_S": 0.010,

    # Catalysis: E_mean (free) converts S -> P with saturation in S
    "kcat": 0.18,
    "K_S": 140.0,

    # Product removal/export
    "d_P": 0.22
}

Y0 = np.array([
    0.3,    # E_mRNA
    6.0,    # E_mean
    20.0,   # E_var
    35.0,   # I_inhibitor
    0.0,    # C_EI
    850.0,  # S_substrate (abundant)
    10.0,   # P_product
    8.0     # R_regulator
], dtype=float)

TSPAN = (0.0, 90.0)

# -------------------------
# (b) ODE function con firma EXACTA
# -------------------------
def dYdt(t, Y):
    p = PARAMS

    mE = max(Y[0], 0.0)
    E  = max(Y[1], 0.0)
    V  = max(Y[2], 0.0)
    I  = max(Y[3], 0.0)
    C  = max(Y[4], 0.0)
    S  = max(Y[5], 0.0)
    P  = max(Y[6], 0.0)
    R  = max(Y[7], 0.0)

    # R -> E transcription (Hill activation)
    act_RE = (R**p["n_RE"]) / (p["K_RE"]**p["n_RE"] + R**p["n_RE"] + 1e-12)
    tx_E = p["k_tx_E_max"] * act_RE

    # Enzyme mean production input (translation)
    u_E = p["k_tl_E"] * mE  # births for E_mean

    # Sequestration: E + I <-> C
    v_on = p["k_on"] * E * I
    v_off = p["k_off"] * C

    # Free enzyme mean decreases by binding; complex increases
    dE_mean = u_E - p["d_E"] * E - v_on + v_off

    # mRNA dynamics (fast)
    dmE = tx_E - p["d_mE"] * mE

    # Variance dynamics (moment approximation for bursty birth-death)
    # dV/dt ≈ F*u_E + d_E*E - 2*d_E*V
    # Keep V nonnegative; this creates realistic noise scaling with production.
    dV = p["F_prod"] * u_E + p["d_E"] * E - 2.0 * p["d_E"] * V

    # Inhibitor dynamics (moderate)
    dI = p["I_prod"] - p["d_I"] * I - v_on + v_off

    # Complex dynamics
    dC = v_on - v_off - p["d_C"] * C

    # Catalysis (deterministic substrate pool) driven by free enzyme mean
    sat_S = S / (p["K_S"] + S + 1e-12)
    v_cat = p["kcat"] * E * sat_S * S  # bounded by saturation; scales with E (low-copy)

    # Substrate/product dynamics (abundant + slow)
    dS = p["S_in"] - p["d_S"] * S - v_cat
    dP = v_cat - p["d_P"] * P

    # Regulator R (slow driver)
    dR = p["k_R_prod"] - p["d_R"] * R

    return np.array([dmE, dE_mean, dV, dI, dC, dS, dP, dR], dtype=float)


# -------------------------
# (c)(d)(e)(f) En __main__
# -------------------------
if __name__ == "__main__":
    t0, tf = TSPAN
    t_span = (t0, tf)
    t_eval = np.linspace(t0, tf, 2000)

    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau")

    if not sol.success:
        print("WARNING: solve_ivp reported failure:", sol.message)

    Y = sol.y  # shape (n_species, n_time)

    # (e) Micro-audit obligatorio
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        traj = Y[i, :]
        q80 = float(np.quantile(traj, 0.80))
        q99 = float(np.quantile(traj, 0.99))
        label = 1 if (q80 < 200.0 and q99 < 200.0) else 0
        labels.append(label)
        print(f"{name}: q80={q80:.3f}, q99={q99:.3f} => label={label}")

    n1 = int(sum(labels))
    n0 = int(len(labels) - n1)
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass labels detected")

    # (f) Plot obligatorio
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(SPECIES_NAMES):
        plt.plot(sol.t, Y[i, :], label=name)
    plt.xlabel("t")
    plt.ylabel("abundance / concentration")
    plt.title(MODEL_NAME)
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_preview.png", dpi=160)
    plt.close()

# -------------------------
# (g) Compatibilidad para el auditor
# -------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
