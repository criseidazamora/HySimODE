# ============================================================================
# Family 6 (Network Motifs / Multiscale) — Incoherent type-1 feed-forward loop
# motif03_i1ffl_saturable_output.py
# I1FFL: X activates Y and Z; Y represses Z.
# Saturable output node: Z_protein drives enzymatic production of P with
# Michaelis–Menten-like saturation and product decay.
# Multiscale: fast mRNAs, slower proteins, abundant substrate pool, and
# low-copy transcription factors.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "motif03_i1ffl_saturable_output"

SPECIES_NAMES = [
    "X_mRNA",        # 0 fast, low
    "X_protein",     # 1 slower, moderate
    "Y_mRNA",        # 2 fast, low
    "Y_protein",     # 3 slower, low-moderate
    "Z_mRNA",        # 4 fast, low
    "Z_protein",     # 5 slower, moderate (can pulse)
    "S_pool",        # 6 abundant substrate pool (metabolite)
    "P_product"      # 7 saturable output product (tracks Z but saturates)
]

PARAMS = {
    # X module
    "k_tx_X": 7.5,
    "d_mX": 3.0,
    "k_tl_X": 6.5,
    "d_X": 0.33,

    # Y activated by X (Hill)
    "k_tx_Y_max": 48.0,
    "K_XY": 42.0,
    "n_XY": 2.0,
    "d_mY": 3.2,
    "k_tl_Y": 7.2,
    "d_Y": 0.55,

    # Z activated by X and repressed by Y (I1FFL)
    "k_tx_Z_max": 85.0,
    "K_XZ": 28.0,
    "n_XZ": 2.0,
    "K_YZ": 16.0,
    "n_YZ": 3.0,
    "d_mZ": 3.4,
    "k_tl_Z": 9.0,
    "d_Z": 0.28,

    # Saturable output node: Z catalyzes S -> P (Michaelis–Menten-like in S, saturable in Z)
    "kcat": 0.85,       # catalytic turnover
    "K_S": 160.0,       # substrate saturation
    "K_Z": 55.0,        # enzyme saturation in Z (prevents unbounded flux at high Z)
    "d_P": 0.20,        # product decay/export
    "d_S": 0.015,       # substrate slow turnover
    "S_in": 14.0        # substrate inflow (keeps S abundant but not constant)
}

Y0 = np.array([
    0.4,     # X_mRNA
    2.0,     # X_protein
    0.2,     # Y_mRNA
    0.6,     # Y_protein
    0.15,    # Z_mRNA
    1.0,     # Z_protein
    900.0,   # S_pool (abundant)
    15.0     # P_product
], dtype=float)

TSPAN = (0.0, 85.0)

# -------------------------
# ODE system
# -------------------------
def dYdt(t, Y):
    p = PARAMS

    X_m = max(Y[0], 0.0)
    X_p = max(Y[1], 0.0)
    Y_m = max(Y[2], 0.0)
    Y_p = max(Y[3], 0.0)
    Z_m = max(Y[4], 0.0)
    Z_p = max(Y[5], 0.0)
    S   = max(Y[6], 0.0)
    P   = max(Y[7], 0.0)

    # Hill activation and repression
    act_XY = (X_p**p["n_XY"]) / (p["K_XY"]**p["n_XY"] + X_p**p["n_XY"] + 1e-12)
    act_XZ = (X_p**p["n_XZ"]) / (p["K_XZ"]**p["n_XZ"] + X_p**p["n_XZ"] + 1e-12)
    rep_YZ = 1.0 / (1.0 + (Y_p / (p["K_YZ"] + 1e-12))**p["n_YZ"])

    # I1FFL transcription
    tx_Y = p["k_tx_Y_max"] * act_XY
    tx_Z = p["k_tx_Z_max"] * act_XZ * rep_YZ

    # Saturable catalytic conversion S -> P by Z (enzyme saturation in Z and substrate saturation in S)
    sat_S = S / (p["K_S"] + S + 1e-12)
    sat_Z = Z_p / (p["K_Z"] + Z_p + 1e-12)
    v_cat = p["kcat"] * sat_Z * sat_S * S  # net flux; bounded by saturations

    # X module
    dX_m = p["k_tx_X"] - p["d_mX"] * X_m
    dX_p = p["k_tl_X"] * X_m - p["d_X"] * X_p

    # Y module
    dY_m = tx_Y - p["d_mY"] * Y_m
    dY_p = p["k_tl_Y"] * Y_m - p["d_Y"] * Y_p

    # Z module
    dZ_m = tx_Z - p["d_mZ"] * Z_m
    dZ_p = p["k_tl_Z"] * Z_m - p["d_Z"] * Z_p

    # Substrate/product module
    dS = p["S_in"] - p["d_S"] * S - v_cat
    dP = v_cat - p["d_P"] * P

    return np.array([dX_m, dX_p, dY_m, dY_p, dZ_m, dZ_p, dS, dP], dtype=float)


# -------------------------
# Main simulation and micro-audit
# -------------------------
if __name__ == "__main__":
    t0, tf = TSPAN
    t_span = (t0, tf)
    t_eval = np.linspace(t0, tf, 2000)

    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau")

    if not sol.success:
        print("WARNING: solve_ivp reported failure:", sol.message)

    Y = sol.y  # shape (n_species, n_time)
  
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

    plt.figure(figsize=(10, 6))
    for i, name in enumerate(SPECIES_NAMES):
        plt.plot(sol.t, Y[i, :], label=name)
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title(f"{MODEL_NAME} (saturable output)")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_preview.png", dpi=160)
    plt.close()

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
