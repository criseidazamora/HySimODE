# signal03_mapk_with_phosphatase.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 2 (Signaling) — MAPK cascade with explicit phosphatase enzyme (PP_inact ⇄ PP_act)
# ============================================================================
MODEL_NAME = "signal03_mapk_with_phosphatase"

SPECIES_NAMES = [
    "Stim",        # S0: upstream stimulus (high-ish copy, decays)
    "KK_inact",    # S1: inactive kinase pool (low copy)
    "KK_act",      # S2: active kinase pool (low copy)
    "PP_inact",    # S3: inactive phosphatase pool (low copy)
    "PP_act",      # S4: active phosphatase pool (low copy)
    "MAPK",        # S5: unphosphorylated MAPK (high copy)
    "MAPKp",       # S6: phosphorylated MAPK (high copy)
    "C_phos",      # S7: KK_act:MAPK complex (low copy)
    "C_dephos",    # S8: PP_act:MAPKp complex (low copy)
]

# ===== PARAMETERS ============================================================
# Units:
#  - first-order rates: /min
#  - bimolecular binding: 1/(molecule*min) in copy-number form
#  - catalytic rates: /min
PARAMS = {
    # Stimulus turnover
    "k_stim_in": 0.0,
    "k_stim_deg": 0.010,

    # Kinase activation/inactivation driven by stimulus
    "k_act": 3.2e-5,      # KK_inact + Stim -> KK_act
    "k_deact": 0.020,     # KK_act -> KK_inact

    # Phosphatase activation/inactivation (explicit phosphatase regulation)
    # (e.g., redox control, inhibitory phosphorylation, or scaffold recruitment)
    "k_pp_act": 6.0e-4,   # PP_inact -> PP_act (/min)
    "k_pp_inact": 0.0025, # PP_act -> PP_inact (/min)
    "stim_boost_pp": 8.0e-5,  # Stim-dependent activation (1/(molecule*min))

    # Phosphorylation: KK_act + MAPK <-> C_phos -> KK_act + MAPKp
    "k_on1": 2.0e-5,
    "k_off1": 0.080,
    "k_cat1": 0.60,

    # Dephosphorylation: PP_act + MAPKp <-> C_dephos -> PP_act + MAPK
    "k_on2": 2.5e-5,
    "k_off2": 0.070,
    "k_cat2": 0.50,

    # Slow MAPK turnover
    "k_syn_mapk": 4.5,     # molecules/min
    "k_deg_mapk": 2.2e-4,  # /min

    # Slow kinase turnover (low-copy)
    "k_syn_kk": 0.22,
    "k_deg_kk": 0.0015,

    # Slow phosphatase turnover (low-copy)
    "k_syn_pp": 0.15,
    "k_deg_pp": 0.0016,
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    3200.0,  # Stim
    140.0,   # KK_inact
    10.0,    # KK_act
    70.0,    # PP_inact
    20.0,    # PP_act
    5200.0,  # MAPK
    200.0,   # MAPKp
    0.0,     # C_phos
    0.0,     # C_dephos
]

TSPAN = (0.0, 1200.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Stim, KKi, KKa, PPi, PPa, MAPK, MAPKp, C1, C2 = Y

    # Stimulus turnover
    v_stim_in = p["k_stim_in"]
    v_stim_deg = p["k_stim_deg"] * Stim

    # Kinase activation/inactivation
    v_act = p["k_act"] * KKi * Stim
    v_deact = p["k_deact"] * KKa

    # Phosphatase activation/inactivation (basal + stimulus-dependent)
    v_pp_act = p["k_pp_act"] * PPi + p["stim_boost_pp"] * PPi * Stim
    v_pp_inact = p["k_pp_inact"] * PPa

    # Phosphorylation complex
    v_on1 = p["k_on1"] * KKa * MAPK
    v_off1 = p["k_off1"] * C1
    v_cat1 = p["k_cat1"] * C1

    # Dephosphorylation complex
    v_on2 = p["k_on2"] * PPa * MAPKp
    v_off2 = p["k_off2"] * C2
    v_cat2 = p["k_cat2"] * C2

    # Protein turnover
    v_syn_mapk = p["k_syn_mapk"]
    v_deg_u = p["k_deg_mapk"] * MAPK
    v_deg_p = p["k_deg_mapk"] * MAPKp

    v_syn_kk = p["k_syn_kk"]
    v_deg_kk_i = p["k_deg_kk"] * KKi
    v_deg_kk_a = p["k_deg_kk"] * KKa

    v_syn_pp = p["k_syn_pp"]
    v_deg_pp_i = p["k_deg_pp"] * PPi
    v_deg_pp_a = p["k_deg_pp"] * PPa

    dStim = v_stim_in - v_stim_deg

    dKKi = v_syn_kk - v_deg_kk_i - v_act + v_deact
    dKKa = -v_deg_kk_a + v_act - v_deact - v_on1 + v_off1 + v_cat1

    dPPi = v_syn_pp - v_deg_pp_i - v_pp_act + v_pp_inact
    dPPa = -v_deg_pp_a + v_pp_act - v_pp_inact - v_on2 + v_off2 + v_cat2

    dMAPK = v_syn_mapk - v_deg_u - v_on1 + v_off1 + v_cat2
    dMAPKp = -v_deg_p + v_cat1 - v_on2 + v_off2

    dC1 = v_on1 - v_off1 - v_cat1
    dC2 = v_on2 - v_off2 - v_cat2

    return np.array([dStim, dKKi, dKKa, dPPi, dPPa, dMAPK, dMAPKp, dC1, dC2], dtype=float)

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
    plt.title("signal03_mapk_with_phosphatase (explicit PP activation state)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal03_mapk_with_phosphatase_preview.png", dpi=220)


var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
