# ============================================================================
# Family 3 (Receptor/Ligand Systems) — Receptor internalization and recycling
# rl03_receptor_internalization.py
# Ligand–receptor binding model with surface activation, differential
# internalization of activated receptor states, endosomal sorting, recycling,
# and degradation of ligand and receptor pools.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "rl03_receptor_internalization"

SPECIES_NAMES = [
    "L",          # S0: extracellular ligand (high copy)
    "R",          # S1: free surface receptor (low copy)
    "LR",         # S2: surface ligand–receptor complex (low-to-mid copy)
    "Rstar",      # S3: activated surface receptor (low copy)
    "LRstar",     # S4: activated ligand-bound receptor (low copy)
    "R_int",      # S5: internalized receptor (low copy)
    "LR_int",     # S6: internalized complex (low copy)
    "R_rec",      # S7: recycling endosome receptor pool (low copy)
    "L_deg",      # S8: degraded/cleared ligand pool (high copy sink)
    "R_deg",      # S9: degraded receptor sink (low copy sink)
]

# ===== PARAMETERS ============================================================
# Units:
#  - bimolecular binding: 1/(molecule*min)
#  - first-order rates: /min
#  - production fluxes: molecules/min
PARAMS = {
    # Ligand supply and extracellular clearance
    "J_L_in": 6500.0,     # molecules/min
    "k_L_clear": 0.004,   # /min

    # Binding/unbinding at surface
    "k_on": 3.0e-6,       # 1/(molecule*min)
    "k_off": 0.030,       # /min

    # Activation steps (surface)
    "k_act_LR": 0.018,    # LR -> LRstar (/min)
    "k_act_R": 0.0015,    # R -> Rstar (/min) basal activation (ligand-independent)

    # Deactivation at surface
    "k_deact_Rstar": 0.006,     # Rstar -> R (/min)
    "k_deact_LRstar": 0.004,    # LRstar -> LR (/min)

    # Internalization (activated internalizes faster)
    "k_int_R": 0.0012,       # R -> R_int (/min)
    "k_int_LR": 0.0025,      # LR -> LR_int (/min)
    "k_int_Rstar": 0.0048,   # Rstar -> R_int (/min)
    "k_int_LRstar": 0.0060,  # LRstar -> LR_int (/min)

    # Sorting from internal pool to recycling compartment
    "k_sort_R": 0.010,       # R_int -> R_rec (/min)
    "k_sort_LR": 0.008,      # LR_int -> R_rec + L_deg (/min) (ligand dissociates/degraded)

    # Recycling to surface
    "k_recycle": 0.006,      # R_rec -> R (/min)

    # Degradation (fraction of internal receptors degraded)
    "k_deg_Rint": 0.0012,    # R_int -> R_deg (/min)
    "k_deg_Rrec": 0.0006,    # R_rec -> R_deg (/min)

    # Receptor synthesis (keeps low-copy receptor pool plausible)
    "k_syn_R": 0.25,         # molecules/min
    "k_deg_Rsurf": 0.0008,   # /min
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    20000.0,  # L
    120.0,    # R
    30.0,     # LR
    10.0,     # Rstar
    8.0,      # LRstar
    20.0,     # R_int
    10.0,     # LR_int
    18.0,     # R_rec
    0.0,      # L_deg
    0.0,      # R_deg
]

TSPAN = (0.0, 2000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    L, R, LR, Rstar, LRstar, Rint, LRint, Rrec, Ldeg, Rdeg = Y

    # Ligand influx and clearance
    v_L_in = p["J_L_in"]
    v_L_clear = p["k_L_clear"] * L

    # Binding/unbinding at surface (R binds ligand; activated species are treated separately)
    v_on = p["k_on"] * L * R
    v_off = p["k_off"] * LR

    # Activation/deactivation at surface
    v_act_LR = p["k_act_LR"] * LR
    v_act_R = p["k_act_R"] * R

    v_deact_Rstar = p["k_deact_Rstar"] * Rstar
    v_deact_LRstar = p["k_deact_LRstar"] * LRstar

    # Internalization (activated internalizes faster)
    v_int_R = p["k_int_R"] * R
    v_int_LR = p["k_int_LR"] * LR
    v_int_Rstar = p["k_int_Rstar"] * Rstar
    v_int_LRstar = p["k_int_LRstar"] * LRstar

    # Sorting to recycling compartment + ligand degradation from internalized complexes
    v_sort_R = p["k_sort_R"] * Rint
    v_sort_LR = p["k_sort_LR"] * LRint

    # Recycling to surface
    v_recycle = p["k_recycle"] * Rrec

    # Degradation
    v_deg_Rint = p["k_deg_Rint"] * Rint
    v_deg_Rrec = p["k_deg_Rrec"] * Rrec
    v_deg_Rsurf = p["k_deg_Rsurf"] * R

    # Receptor synthesis
    v_syn_R = p["k_syn_R"]

    dL = v_L_in - v_L_clear - v_on + v_off
    dR = v_syn_R - v_deg_Rsurf - v_on + v_off - v_act_R + v_deact_Rstar - v_int_R + v_recycle
    dLR = v_on - v_off - v_act_LR + v_deact_LRstar - v_int_LR
    dRstar = v_act_R - v_deact_Rstar - v_int_Rstar
    dLRstar = v_act_LR - v_deact_LRstar - v_int_LRstar

    # Internal pools: receive internalization, lose by sorting and degradation
    dRint = v_int_R + v_int_Rstar - v_sort_R - v_deg_Rint
    dLRint = v_int_LR + v_int_LRstar - v_sort_LR

    # Recycling compartment: receives sorted receptors (including from LRint after ligand is removed),
    # loses by recycling and degradation
    dRrec = v_sort_R + v_sort_LR - v_recycle - v_deg_Rrec

    # Degraded pools
    dLdeg = v_L_clear + v_sort_LR
    dRdeg = v_deg_Rint + v_deg_Rrec + v_deg_Rsurf

    return np.array([dL, dR, dLR, dRstar, dLRstar, dRint, dLRint, dRrec, dLdeg, dRdeg], dtype=float)

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
    plt.title("rl03_receptor_internalization (activated receptor internalization/recycling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rl03_receptor_internalization_preview.png", dpi=220)


var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
