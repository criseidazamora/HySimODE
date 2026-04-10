# ============================================================================
# Family 3 (Receptor/Ligand Systems) — Ligand–receptor binding with trafficking
# rl01_ligand_receptor_binding.py
# Ligand–receptor association/dissociation model with receptor insertion,
# internalization, recycling, and ligand clearance driven by a decaying
# secretion pulse.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "rl01_ligand_receptor_binding"

SPECIES_NAMES = [
    "L",        # S0: extracellular ligand (high copy)
    "R",        # S1: free receptor (low copy)
    "LR",       # S2: ligand–receptor complex (low-to-mid copy)
    "R_totBuf", # S3: receptor reserve/buffer pool (low copy; slow insertion/removal)
    "L_deg",    # S4: degraded/cleared ligand pool (high copy sink)
    "R_int",    # S5: internalized receptor (low copy)
    "LR_int",   # S6: internalized complex (low copy)
    "L_prod",   # S7: ligand production driver (high copy; decays)
]

# ===== PARAMETERS ============================================================
# Units:
#  - bimolecular binding: 1/(molecule*min) in copy-number form
#  - first-order rates: /min
#  - production fluxes: molecules/min
PARAMS = {
    # Ligand production driver (decaying secretion pulse)
    "k_Lprod_in": 0.0,     # no ongoing input; initial L_prod decays
    "k_Lprod_deg": 0.010,  # /min
    "k_sec": 6.0,          # secretion gain: L_prod -> L (molecules/min per L_prod unit)

    # Ligand clearance (extracellular)
    "k_L_clear": 0.004,    # /min

    # Receptor insertion/removal (membrane trafficking, slow)
    "k_R_ins": 0.006,      # R_totBuf -> R (/min)
    "k_R_rem": 0.003,      # R -> R_totBuf (/min)

    # Ligand–receptor binding
    "k_on": 3.0e-6,        # 1/(molecule*min)
    "k_off": 0.030,        # /min

    # Basal internalization (slow, plausible)
    "k_int_R": 0.0012,     # R -> R_int (/min)
    "k_int_LR": 0.0030,    # LR -> LR_int (/min)

    # Recycling (endosomal return)
    "k_rec_R": 0.0020,     # R_int -> R (/min)
    "k_rec_LR": 0.0008,    # LR_int -> LR (/min)

    # Degradation in endosomes/lysosomes
    "k_deg_Rint": 0.0009,  # R_int -> loss (/min) (implicit sink)
    "k_deg_LRint": 0.0015, # LR_int -> L_deg (+ receptor loss implicit) (/min)

    # Ligand release from internalized complex contributing to degraded ligand pool
    "frac_L_deg": 1.0,     # fraction of LR_int degradation counted in L_deg
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    25000.0,  # L
    120.0,    # R
    30.0,     # LR
    80.0,     # R_totBuf
    0.0,      # L_deg
    20.0,     # R_int
    10.0,     # LR_int
    1200.0,   # L_prod
]

TSPAN = (0.0, 2000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    L, R, LR, Rbuf, Ldeg, Rint, LRint, Lprod = Y

    # Ligand production driver decays (pulse-like secretion)
    v_Lprod_in = p["k_Lprod_in"]
    v_Lprod_deg = p["k_Lprod_deg"] * Lprod
    v_sec = p["k_sec"] * Lprod

    # Extracellular ligand clearance into degraded pool
    v_L_clear = p["k_L_clear"] * L

    # Receptor insertion/removal
    v_R_ins = p["k_R_ins"] * Rbuf
    v_R_rem = p["k_R_rem"] * R

    # Binding/unbinding at the membrane
    v_on = p["k_on"] * L * R
    v_off = p["k_off"] * LR

    # Internalization and recycling
    v_int_R = p["k_int_R"] * R
    v_int_LR = p["k_int_LR"] * LR
    v_rec_R = p["k_rec_R"] * Rint
    v_rec_LR = p["k_rec_LR"] * LRint

    # Degradation (implicit receptor loss; ligand counted in L_deg from LRint)
    v_deg_Rint = p["k_deg_Rint"] * Rint
    v_deg_LRint = p["k_deg_LRint"] * LRint

    dLprod = v_Lprod_in - v_Lprod_deg

    dL = v_sec - v_L_clear - v_on + v_off
    dR = v_R_ins - v_R_rem - v_on + v_off - v_int_R + v_rec_R
    dLR = v_on - v_off - v_int_LR + v_rec_LR

    dRbuf = -v_R_ins + v_R_rem
    dRint = v_int_R - v_rec_R - v_deg_Rint
    dLRint = v_int_LR - v_rec_LR - v_deg_LRint

    dLdeg = v_L_clear + p["frac_L_deg"] * v_deg_LRint

    return np.array([dL, dR, dLR, dRbuf, dLdeg, dRint, dLRint, dLprod], dtype=float)

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

    # Preview plot of species trajectories
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("rl01_ligand_receptor_binding (association/dissociation with slow trafficking)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rl01_ligand_receptor_binding_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
