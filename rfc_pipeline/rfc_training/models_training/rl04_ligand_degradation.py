# ============================================================================
# Family 3 (Receptor/Ligand Systems) — Ligand degradation with endocytic clearance
# rl04_ligand_degradation.py
# Ligand–receptor binding model with ligand production, endocytosis, receptor
# recycling, lysosomal degradation of internalized ligand, and saturable
# extracellular clearance mediated by a low-copy scavenger.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "rl04_ligand_degradation"

SPECIES_NAMES = [
    "L",          # S0: extracellular ligand (high copy)
    "R",          # S1: surface receptor (low copy)
    "LR",         # S2: surface complex (low-to-mid copy)
    "L_int",      # S3: internalized ligand (high-to-mid copy)
    "LR_int",     # S4: internalized complex (low copy)
    "R_int",      # S5: internalized receptor (low copy)
    "L_deg",      # S6: degraded ligand pool (high copy sink)
    "Prod",       # S7: ligand production driver (high copy; slow fluctuations)
    "Clear",      # S8: clearance capacity / scavenger (low copy)
]

# ===== PARAMETERS ============================================================
# Units:
#  - bimolecular binding: 1/(molecule*min)
#  - first-order rates: /min
#  - production fluxes: molecules/min
PARAMS = {
    # Ligand production driver (slow approach to baseline)
    "Prod_base": 1400.0,   # baseline driver level (molecules)
    "k_prod_relax": 0.003, # /min relaxation toward baseline
    "k_sec": 5.5,          # secretion gain: Prod -> L (molecules/min per Prod unit)

    # Receptor binding/unbinding
    "k_on": 2.8e-6,        # 1/(molecule*min)
    "k_off": 0.028,        # /min

    # Internalization
    "k_int_L": 0.0015,     # L -> L_int (/min) (fluid-phase endocytosis)
    "k_int_LR": 0.0035,    # LR -> LR_int (/min)
    "k_int_R": 0.0010,     # R -> R_int (/min)

    # Recycling
    "k_rec_L": 0.0008,     # L_int -> L (/min)
    "k_rec_LR": 0.0006,    # LR_int -> LR (/min)
    "k_rec_R": 0.0018,     # R_int -> R (/min)

    # Lysosomal degradation (ligand-focused)
    "k_deg_Lint": 0.0030,  # L_int -> L_deg (/min)
    "k_deg_LRint": 0.0022, # LR_int -> L_deg (+ receptor loss implicit) (/min)

    # Receptor degradation (slow)
    "k_deg_Rint": 0.0010,  # R_int -> loss (/min)

    # Surface receptor turnover
    "k_syn_R": 0.22,       # molecules/min
    "k_deg_R": 0.0009,     # /min

    # Extracellular clearance by scavenger (saturable in L, capacity low-copy)
    "k_clear": 0.020,      # /min (per scavenger unit)
    "K_clear": 12000.0,    # molecules

    # Scavenger turnover (low-copy)
    "k_syn_clear": 0.10,   # molecules/min
    "k_deg_clear": 0.0020, # /min
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    18000.0,  # L
    110.0,    # R
    25.0,     # LR
    1200.0,   # L_int
    8.0,      # LR_int
    18.0,     # R_int
    0.0,      # L_deg
    1100.0,   # Prod
    60.0,     # Clear
]

TSPAN = (0.0, 2200.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    L, R, LR, Lint, LRint, Rint, Ldeg, Prod, Clear = Y

    # Production driver relaxes toward baseline (slow drift)
    v_prod_relax = p["k_prod_relax"] * (p["Prod_base"] - Prod)
    v_sec = p["k_sec"] * Prod

    # Binding/unbinding
    v_on = p["k_on"] * L * R
    v_off = p["k_off"] * LR

    # Internalization
    v_int_L = p["k_int_L"] * L
    v_int_LR = p["k_int_LR"] * LR
    v_int_R = p["k_int_R"] * R

    # Recycling
    v_rec_L = p["k_rec_L"] * Lint
    v_rec_LR = p["k_rec_LR"] * LRint
    v_rec_R = p["k_rec_R"] * Rint

    # Degradation
    v_deg_Lint = p["k_deg_Lint"] * Lint
    v_deg_LRint = p["k_deg_LRint"] * LRint
    v_deg_Rint = p["k_deg_Rint"] * Rint

    # Extracellular clearance by scavenger (saturable in L; capacity set by low-copy Clear)
    sat_clear = L / (p["K_clear"] + L)
    v_clear = p["k_clear"] * Clear * sat_clear * L  # effective uptake/clearance

    # Receptor turnover at surface
    v_syn_R = p["k_syn_R"]
    v_deg_R = p["k_deg_R"] * R

    # Scavenger turnover
    v_syn_clear = p["k_syn_clear"]
    v_deg_clear = p["k_deg_clear"] * Clear

    dProd = v_prod_relax

    dL = v_sec - v_on + v_off - v_int_L + v_rec_L - v_clear
    dR = v_syn_R - v_deg_R - v_on + v_off - v_int_R + v_rec_R
    dLR = v_on - v_off - v_int_LR + v_rec_LR

    dLint = v_int_L - v_rec_L - v_deg_Lint
    dLRint = v_int_LR - v_rec_LR - v_deg_LRint
    dRint = v_int_R - v_rec_R - v_deg_Rint

    dLdeg = v_deg_Lint + v_deg_LRint + v_clear
    dClear = v_syn_clear - v_deg_clear

    return np.array([dL, dR, dLR, dLint, dLRint, dRint, dLdeg, dProd, dClear], dtype=float)

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
    plt.title("rl04_ligand_degradation (production, endocytosis, and saturable clearance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rl04_ligand_degradation_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
