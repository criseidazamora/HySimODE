# signal01_mapk_single_layer.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 2 (Signaling) — Single-layer MAPK phosphorylation–dephosphorylation cycle
# ============================================================================
MODEL_NAME = "signal01_mapk_single_layer"

SPECIES_NAMES = [
    "Stim",       # S0: upstream stimulus (high-ish copy, decays)
    "KK_inact",   # S1: inactive kinase pool (low copy)
    "KK_act",     # S2: active kinase pool (low copy)
    "PPase",      # S3: phosphatase pool (low copy)
    "MAPK",       # S4: unphosphorylated MAPK substrate (high copy)
    "MAPKp",      # S5: phosphorylated MAPK (high copy)
    "C_phos",     # S6: KK_act:MAPK complex (low copy)
    "C_dephos",   # S7: PPase:MAPKp complex (low copy)
]

# ===== PARAMETERS ============================================================
# Units:
#  - first-order rates: /min
#  - bimolecular binding: 1/(molecule*min) in copy-number form
#  - catalytic rates: /min
PARAMS = {
    # Stimulus turnover (creates early vs late behavior)
    "k_stim_in": 0.0,     # no ongoing input; initial Stim decays
    "k_stim_deg": 0.010,  # /min

    # Kinase activation/inactivation driven by stimulus
    "k_act": 3.5e-5,      # KK_inact + Stim -> KK_act  (1/(molecule*min))
    "k_deact": 0.020,     # KK_act -> KK_inact (/min)

    # Phosphorylation module: KK_act + MAPK <-> C_phos -> KK_act + MAPKp
    "k_on1": 2.0e-5,      # binding (1/(molecule*min))
    "k_off1": 0.080,      # unbinding (/min)
    "k_cat1": 0.60,       # catalysis (/min)

    # Dephosphorylation module: PPase + MAPKp <-> C_dephos -> PPase + MAPK
    "k_on2": 2.5e-5,      # binding (1/(molecule*min))
    "k_off2": 0.070,      # unbinding (/min)
    "k_cat2": 0.45,       # catalysis (/min)

    # Slow MAPK turnover (keeps biology plausible, avoids strict conservation lock-in)
    "k_syn_mapk": 5.0,     # molecules/min
    "k_deg_mapk": 2.5e-4,  # /min

    # Slow enzyme turnover (low-copy)
    "k_syn_kk": 0.25,      # molecules/min
    "k_deg_kk": 0.0015,    # /min
    "k_syn_pp": 0.18,      # molecules/min
    "k_deg_pp": 0.0018,    # /min
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    3000.0,  # Stim
    140.0,   # KK_inact
    10.0,    # KK_act
    85.0,    # PPase
    5200.0,  # MAPK
    200.0,   # MAPKp
    0.0,     # C_phos
    0.0,     # C_dephos
]

TSPAN = (0.0, 1200.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Stim, KKi, KKa, PP, MAPK, MAPKp, C1, C2 = Y

    # Stimulus turnover (decay after initial pulse)
    v_stim_in = p["k_stim_in"]
    v_stim_deg = p["k_stim_deg"] * Stim

    # Kinase activation by stimulus
    v_act = p["k_act"] * KKi * Stim
    v_deact = p["k_deact"] * KKa

    # Phosphorylation binding/unbinding/catalysis
    v_on1 = p["k_on1"] * KKa * MAPK
    v_off1 = p["k_off1"] * C1
    v_cat1 = p["k_cat1"] * C1

    # Dephosphorylation binding/unbinding/catalysis
    v_on2 = p["k_on2"] * PP * MAPKp
    v_off2 = p["k_off2"] * C2
    v_cat2 = p["k_cat2"] * C2

    # Slow synthesis/degradation
    v_syn_mapk = p["k_syn_mapk"]
    v_deg_mapk_u = p["k_deg_mapk"] * MAPK
    v_deg_mapk_p = p["k_deg_mapk"] * MAPKp

    v_syn_kk = p["k_syn_kk"]
    v_deg_kk_i = p["k_deg_kk"] * KKi
    v_deg_kk_a = p["k_deg_kk"] * KKa

    v_syn_pp = p["k_syn_pp"]
    v_deg_pp = p["k_deg_pp"] * PP

    dStim = v_stim_in - v_stim_deg

    dKKi = v_syn_kk - v_deg_kk_i - v_act + v_deact
    dKKa = -v_deg_kk_a + v_act - v_deact - v_on1 + v_off1 + v_cat1

    dPP = v_syn_pp - v_deg_pp - v_on2 + v_off2 + v_cat2

    dMAPK = v_syn_mapk - v_deg_mapk_u - v_on1 + v_off1 + v_cat2
    dMAPKp = -v_deg_mapk_p + v_cat1 - v_on2 + v_off2

    dC1 = v_on1 - v_off1 - v_cat1
    dC2 = v_on2 - v_off2 - v_cat2

    return np.array([dStim, dKKi, dKKa, dPP, dMAPK, dMAPKp, dC1, dC2], dtype=float)

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

    # Optional plot (preview)
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("signal01_mapk_single_layer (stimulus-driven kinase, MAPK phosphorylation cycle)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal01_mapk_single_layer_preview.png", dpi=220)
    # ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------

# The auditor expects this variable name:
var_names = SPECIES_NAMES

# The auditor expects an ODE function whose name ends with "_odes".
# This is just a thin wrapper around your standard dYdt(t, Y).
def model_odes(t, y):
    return dYdt(t, y)

