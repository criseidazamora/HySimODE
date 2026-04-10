# ============================================================================
# Family 2 (Signaling) — MAPK double phosphorylation cycle
# signal02_mapk_double_phosphorylation.py
# Stimulus-driven kinase activation coupled to sequential MAPK phosphorylation
# (MAPK -> MAPKp -> MAPKpp) and two-step phosphatase-mediated dephosphorylation,
# with explicit enzyme–substrate complexes.
# ============================================================================
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "signal02_mapk_double_phosphorylation"

SPECIES_NAMES = [
    "Stim",        # S0: upstream stimulus (high-ish copy, decays)
    "KK_inact",    # S1: inactive kinase pool (low copy)
    "KK_act",      # S2: active kinase pool (low copy)
    "PPase",       # S3: phosphatase pool (low copy)
    "MAPK",        # S4: unphosphorylated substrate (high copy)
    "MAPKp",       # S5: singly phosphorylated (high copy)
    "MAPKpp",      # S6: doubly phosphorylated (high copy)
    "C_phos1",     # S7: KK_act:MAPK complex (low copy)
    "C_phos2",     # S8: KK_act:MAPKp complex (low copy)
    "C_dephos1",   # S9: PPase:MAPKp complex (low copy)
    "C_dephos2",   # S10: PPase:MAPKpp complex (low copy)
]

# ===== PARAMETERS ============================================================
# Units:
#  - first-order rates: /min
#  - bimolecular binding: 1/(molecule*min) in copy-number form
#  - catalytic rates: /min
PARAMS = {
    # Stimulus turnover (early vs late behavior)
    "k_stim_in": 0.0,     # no ongoing input
    "k_stim_deg": 0.010,  # /min

    # Kinase activation/inactivation driven by stimulus
    "k_act": 3.0e-5,      # KK_inact + Stim -> KK_act  (1/(molecule*min))
    "k_deact": 0.020,     # KK_act -> KK_inact (/min)

    # Phosphorylation step 1: KK_act + MAPK <-> C_phos1 -> KK_act + MAPKp
    "k_on1": 2.0e-5,
    "k_off1": 0.080,
    "k_cat1": 0.55,

    # Phosphorylation step 2: KK_act + MAPKp <-> C_phos2 -> KK_act + MAPKpp
    "k_on2": 2.0e-5,
    "k_off2": 0.070,
    "k_cat2": 0.50,

    # Dephosphorylation step 1: PPase + MAPKp <-> C_dephos1 -> PPase + MAPK
    "k_on3": 2.5e-5,
    "k_off3": 0.070,
    "k_cat3": 0.45,

    # Dephosphorylation step 2: PPase + MAPKpp <-> C_dephos2 -> PPase + MAPKp
    "k_on4": 2.5e-5,
    "k_off4": 0.060,
    "k_cat4": 0.40,

    # Slow MAPK turnover (plausible protein turnover)
    "k_syn_mapk": 5.0,     # molecules/min
    "k_deg_mapk": 2.5e-4,  # /min

    # Slow enzyme turnover (low-copy)
    "k_syn_kk": 0.22,      # molecules/min
    "k_deg_kk": 0.0015,    # /min
    "k_syn_pp": 0.18,      # molecules/min
    "k_deg_pp": 0.0018,    # /min
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    3200.0,  # Stim
    150.0,   # KK_inact
    8.0,     # KK_act
    95.0,    # PPase
    5600.0,  # MAPK
    250.0,   # MAPKp
    40.0,    # MAPKpp
    0.0,     # C_phos1
    0.0,     # C_phos2
    0.0,     # C_dephos1
    0.0,     # C_dephos2
]

TSPAN = (0.0, 1200.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Stim, KKi, KKa, PP, MAPK, MAPKp, MAPKpp, C1, C2, C3, C4 = Y

    # Stimulus turnover
    v_stim_in = p["k_stim_in"]
    v_stim_deg = p["k_stim_deg"] * Stim

    # Kinase activation/inactivation by stimulus
    v_act = p["k_act"] * KKi * Stim
    v_deact = p["k_deact"] * KKa

    # Phosphorylation step 1
    v_on1 = p["k_on1"] * KKa * MAPK
    v_off1 = p["k_off1"] * C1
    v_cat1 = p["k_cat1"] * C1

    # Phosphorylation step 2
    v_on2 = p["k_on2"] * KKa * MAPKp
    v_off2 = p["k_off2"] * C2
    v_cat2 = p["k_cat2"] * C2

    # Dephosphorylation step 1
    v_on3 = p["k_on3"] * PP * MAPKp
    v_off3 = p["k_off3"] * C3
    v_cat3 = p["k_cat3"] * C3

    # Dephosphorylation step 2
    v_on4 = p["k_on4"] * PP * MAPKpp
    v_off4 = p["k_off4"] * C4
    v_cat4 = p["k_cat4"] * C4

    # Slow synthesis/degradation (MAPK synthesized into unphosphorylated form)
    v_syn_mapk = p["k_syn_mapk"]
    v_deg_u = p["k_deg_mapk"] * MAPK
    v_deg_p = p["k_deg_mapk"] * MAPKp
    v_deg_pp = p["k_deg_mapk"] * MAPKpp

    v_syn_kk = p["k_syn_kk"]
    v_deg_kk_i = p["k_deg_kk"] * KKi
    v_deg_kk_a = p["k_deg_kk"] * KKa

    v_syn_pp = p["k_syn_pp"]
    v_deg_ppase = p["k_deg_pp"] * PP

    dStim = v_stim_in - v_stim_deg

    dKKi = v_syn_kk - v_deg_kk_i - v_act + v_deact
    dKKa = -v_deg_kk_a + v_act - v_deact - v_on1 + v_off1 + v_cat1 - v_on2 + v_off2 + v_cat2

    dPP = v_syn_pp - v_deg_ppase - v_on3 + v_off3 + v_cat3 - v_on4 + v_off4 + v_cat4

    dMAPK = v_syn_mapk - v_deg_u - v_on1 + v_off1 + v_cat3
    dMAPKp = -v_deg_p + v_cat1 - v_on2 + v_off2 + v_cat4 - v_on3 + v_off3
    dMAPKpp = -v_deg_pp + v_cat2 - v_on4 + v_off4

    dC1 = v_on1 - v_off1 - v_cat1
    dC2 = v_on2 - v_off2 - v_cat2
    dC3 = v_on3 - v_off3 - v_cat3
    dC4 = v_on4 - v_off4 - v_cat4

    return np.array([dStim, dKKi, dKKa, dPP, dMAPK, dMAPKp, dMAPKpp, dC1, dC2, dC3, dC4], dtype=float)

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
    plt.title("signal02_mapk_double_phosphorylation (MAPK → MAPKp → MAPKpp)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal02_mapk_double_phosphorylation_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
