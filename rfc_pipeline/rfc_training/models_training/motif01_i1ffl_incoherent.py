# ============================================================================
# Family 6 (Network Motifs / Multiscale) — Incoherent type-1 feed-forward loop
# motif01_i1ffl_incoherent.py
# Incoherent type-1 feed-forward loop (I1FFL): X activates Y and Z; Y represses Z.
# Multiscale: fast mRNA-like species, slower proteins/complexes; abundant
# substrate pool versus scarce transcription factor.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "motif01_i1ffl_incoherent"

SPECIES_NAMES = [
    "X_mRNA",        # 0 fast, low copy
    "X_protein",     # 1 slower, moderate
    "Y_mRNA",        # 2 fast, low
    "Y_protein",     # 3 slower, low-moderate
    "Z_mRNA",        # 4 fast, low
    "Z_protein",     # 5 slower, moderate
    "W_buffer",      # 6 abundant buffering pool (metabolite/chaperone-like)
    "C_YW_complex"   # 7 complex that sequesters Y (buffering / adaptation shaping)
]

PARAMS = {
    # X transcription & translation
    "k_tx_X": 8.0,        # basal transcription of X mRNA
    "d_mX": 3.0,          # fast mRNA degradation
    "k_tl_X": 6.0,        # translation rate
    "d_X": 0.35,          # protein degradation/dilution

    # Y transcription activated by X_protein (Hill)
    "k_tx_Y_max": 40.0,
    "K_XY": 40.0,
    "n_XY": 2.0,
    "d_mY": 3.2,
    "k_tl_Y": 7.5,
    "d_Y": 0.55,

    # Z transcription activated by X_protein but repressed by Y_protein (I1FFL)
    "k_tx_Z_max": 65.0,
    "K_XZ": 30.0,
    "n_XZ": 2.0,
    "K_YZ": 18.0,
    "n_YZ": 3.0,
    "d_mZ": 3.4,
    "k_tl_Z": 8.5,
    "d_Z": 0.28,

    # Buffer pool W and complex formation with Y
    "W_prod": 40.0,       # maintains abundant W
    "d_W": 0.02,          # very slow turnover (abundant)
    "k_on": 0.06,         # association
    "k_off": 1.0,         # dissociation (fast-ish)
    "d_C": 0.05,          # slow complex clearance

    # Nonlinear consumption of Z by W (buffering/clearance term to add adaptation/overshoot)
    "k_consume": 0.0012,
    "K_consume": 120.0
}

Y0 = np.array([
    0.5,    # X_mRNA
    2.0,    # X_protein
    0.2,    # Y_mRNA
    0.8,    # Y_protein
    0.2,    # Z_mRNA
    1.0,    # Z_protein
    600.0,  # W_buffer (abundant)
    0.0     # C_YW_complex
], dtype=float)

TSPAN = (0.0, 80.0)

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
    W   = max(Y[6], 0.0)
    C   = max(Y[7], 0.0)

    # Hill activation for Y by X_protein
    act_XY = (X_p**p["n_XY"]) / (p["K_XY"]**p["n_XY"] + X_p**p["n_XY"] + 1e-12)

    # I1FFL: Z activated by X, repressed by Y (multiplicative)
    act_XZ = (X_p**p["n_XZ"]) / (p["K_XZ"]**p["n_XZ"] + X_p**p["n_XZ"] + 1e-12)
    rep_YZ = 1.0 / (1.0 + (Y_p / (p["K_YZ"] + 1e-12))**p["n_YZ"])

    # Nonlinear consumption/clearance of Z by W (saturable)
    consume_Z = p["k_consume"] * W * (Z_p / (p["K_consume"] + Z_p + 1e-12))

    # Complex formation Y + W <-> C (Y sequestration by abundant buffer)
    v_on  = p["k_on"]  * Y_p * W
    v_off = p["k_off"] * C

    # X module
    dX_m = p["k_tx_X"] - p["d_mX"] * X_m
    dX_p = p["k_tl_X"] * X_m - p["d_X"] * X_p

    # Y module
    tx_Y = p["k_tx_Y_max"] * act_XY
    dY_m = tx_Y - p["d_mY"] * Y_m
    dY_p = p["k_tl_Y"] * Y_m - p["d_Y"] * Y_p - v_on + v_off  # sequestration lowers free Y_p

    # Z module (I1FFL produces pulse/adaptation: early X activates Z, later Y represses)
    tx_Z = p["k_tx_Z_max"] * act_XZ * rep_YZ
    dZ_m = tx_Z - p["d_mZ"] * Z_m
    dZ_p = p["k_tl_Z"] * Z_m - p["d_Z"] * Z_p - consume_Z

    # Buffer W: maintained abundant but can be drawn into complex; slow turnover
    dW = p["W_prod"] - p["d_W"] * W - v_on + v_off

    # Complex
    dC = v_on - v_off - p["d_C"] * C

    return np.array([dX_m, dX_p, dY_m, dY_p, dZ_m, dZ_p, dW, dC], dtype=float)


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
    plt.title("motif01_i1ffl_incoherent (I1FFL)")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_preview.png", dpi=160)
    plt.close()

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
