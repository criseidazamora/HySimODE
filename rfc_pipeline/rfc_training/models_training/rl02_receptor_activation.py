# ============================================================================
# Family 3 (Receptor/Ligand Systems) — Receptor activation with desensitization
# rl02_receptor_activation.py
# Ligand–receptor binding model with activation of an active receptor state,
# followed by desensitization, internalization, recycling, and ligand clearance.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "rl02_receptor_activation"

SPECIES_NAMES = [
    "L",        # S0: extracellular ligand (high copy)
    "R",        # S1: free receptor (low copy)
    "LR",       # S2: ligand–receptor complex (low-to-mid copy)
    "Rstar",    # S3: activated receptor state (low copy)
    "RstarI",   # S4: inactivated/desensitized receptor (low copy)
    "R_int",    # S5: internalized free receptor (low copy)
    "LR_int",   # S6: internalized complex (low copy)
    "L_deg",    # S7: degraded/cleared ligand pool (high copy sink)
    "R_res",    # S8: receptor reserve pool (low copy; insertion/removal)
]

# ===== PARAMETERS ============================================================
# Units:
#  - bimolecular binding: 1/(molecule*min)
#  - first-order rates: /min
#  - production fluxes: molecules/min
PARAMS = {
    # Ligand supply and clearance
    "J_L_in": 7000.0,     # constant ligand influx (molecules/min)
    "k_L_clear": 0.004,   # extracellular clearance (/min)

    # Receptor insertion/removal (slow trafficking)
    "k_R_ins": 0.006,     # R_res -> R (/min)
    "k_R_rem": 0.003,     # R -> R_res (/min)

    # Binding/unbinding
    "k_on": 3.2e-6,       # 1/(molecule*min)
    "k_off": 0.028,       # /min

    # Activation: LR -> Rstar + L (ligand-catalyzed conformational activation with ligand release)
    "k_act": 0.020,       # /min

    # Active receptor inactivation/desensitization
    "k_inact": 0.010,     # Rstar -> RstarI (/min)
    "k_recover": 0.0020,  # RstarI -> R (/min)

    # Internalization and recycling (state-dependent; active tends to internalize faster)
    "k_int_R": 0.0010,    # R -> R_int (/min)
    "k_int_LR": 0.0028,   # LR -> LR_int (/min)
    "k_int_Rstar": 0.0045,# Rstar -> R_int (/min)
    "k_rec_R": 0.0022,    # R_int -> R (/min)
    "k_deg_Rint": 0.0010, # R_int -> loss (/min)

    # Internalized complex fates
    "k_rec_LR": 0.0008,   # LR_int -> LR (/min)
    "k_deg_LRint": 0.0016,# LR_int -> L_deg (+ receptor loss implicit) (/min)

    # Ligand sink accounting
    "frac_L_deg": 1.0,
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    22000.0,  # L
    110.0,    # R
    25.0,     # LR
    15.0,     # Rstar
    10.0,     # RstarI
    15.0,     # R_int
    8.0,      # LR_int
    0.0,      # L_deg
    90.0,     # R_res
]

TSPAN = (0.0, 2000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    L, R, LR, Rstar, RstarI, Rint, LRint, Ldeg, Rres = Y

    # Ligand influx and clearance
    v_L_in = p["J_L_in"]
    v_L_clear = p["k_L_clear"] * L

    # Receptor insertion/removal
    v_R_ins = p["k_R_ins"] * Rres
    v_R_rem = p["k_R_rem"] * R

    # Binding/unbinding
    v_on = p["k_on"] * L * R
    v_off = p["k_off"] * LR

    # Activation: complex generates active receptor, ligand released back to extracellular pool
    v_act = p["k_act"] * LR

    # Active receptor inactivation and recovery
    v_inact = p["k_inact"] * Rstar
    v_recover = p["k_recover"] * RstarI

    # Internalization/recycling/degradation
    v_int_R = p["k_int_R"] * R
    v_int_LR = p["k_int_LR"] * LR
    v_int_Rstar = p["k_int_Rstar"] * Rstar

    v_rec_R = p["k_rec_R"] * Rint
    v_deg_Rint = p["k_deg_Rint"] * Rint

    v_rec_LR = p["k_rec_LR"] * LRint
    v_deg_LRint = p["k_deg_LRint"] * LRint

    # ODEs
    dL = v_L_in - v_L_clear - v_on + v_off + v_act
    dR = v_R_ins - v_R_rem - v_on + v_off + v_recover - v_int_R + v_rec_R
    dLR = v_on - v_off - v_act - v_int_LR + v_rec_LR
    dRstar = v_act - v_inact - v_int_Rstar
    dRstarI = v_inact - v_recover
    dRint = v_int_R + v_int_Rstar - v_rec_R - v_deg_Rint
    dLRint = v_int_LR - v_rec_LR - v_deg_LRint
    dLdeg = v_L_clear + p["frac_L_deg"] * v_deg_LRint
    dRres = -v_R_ins + v_R_rem

    return np.array([dL, dR, dLR, dRstar, dRstarI, dRint, dLRint, dLdeg, dRres], dtype=float)

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
    plt.title("rl02_receptor_activation (LR→R* activation with desensitization/trafficking)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rl02_receptor_activation_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)

