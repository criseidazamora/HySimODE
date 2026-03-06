# signal05_mapk_three_layer.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 2 (Signaling) — Three-layer MAPK cascade (Raf–MEK–ERK simplified)
# ============================================================================
MODEL_NAME = "signal05_mapk_three_layer"

SPECIES_NAMES = [
    "Stim",       # S0: upstream stimulus (high-ish copy, decays)
    "Raf_inact",  # S1: inactive Raf pool (low copy)
    "Raf_act",    # S2: active Raf pool (low copy)
    "MEK",        # S3: unphosphorylated MEK (high copy)
    "MEKp",       # S4: phosphorylated MEK (high copy)
    "ERK",        # S5: unphosphorylated ERK (high copy)
    "ERKp",       # S6: phosphorylated ERK (high copy)
    "PP_MEK",     # S7: MEK phosphatase (low copy)
    "PP_ERK",     # S8: ERK phosphatase (low copy)
]

# ===== PARAMETERS ============================================================
# Units:
#  - first-order rates: /min
#  - bimolecular activation: 1/(molecule*min) in copy-number form
#  - MM parameters: kcat (/min), Km (molecules)
PARAMS = {
    # Stimulus turnover (early vs late)
    "k_stim_in": 0.0,
    "k_stim_deg": 0.008,   # /min

    # Raf activation/inactivation
    "k_raf_act": 2.8e-5,   # Raf_inact + Stim -> Raf_act (1/(molecule*min))
    "k_raf_deact": 0.018,  # Raf_act -> Raf_inact (/min)

    # Raf -> MEK phosphorylation (MM with explicit Raf_act catalyst)
    "kcat_mek": 1.6,       # /min
    "Km_mek": 1400.0,      # molecules

    # MEKp -> ERK phosphorylation (MM with explicit MEKp catalyst)
    "kcat_erk": 1.2,       # /min
    "Km_erk": 1800.0,      # molecules

    # Phosphatase reactions (MM)
    "kcat_pp_mek": 1.0,    # /min
    "Km_pp_mek": 1100.0,   # molecules
    "kcat_pp_erk": 0.9,    # /min
    "Km_pp_erk": 1300.0,   # molecules

    # Slow protein turnover (plausible, avoids rigid conservation)
    "k_syn_mek": 2.8,      # molecules/min
    "k_deg_mek": 2.0e-4,   # /min
    "k_syn_erk": 3.2,      # molecules/min
    "k_deg_erk": 2.0e-4,   # /min

    # Enzyme turnover (low-copy)
    "k_syn_raf": 0.18,     # molecules/min
    "k_deg_raf": 0.0014,   # /min
    "k_syn_pp_mek": 0.12,  # molecules/min
    "k_deg_pp_mek": 0.0017,# /min
    "k_syn_pp_erk": 0.10,  # molecules/min
    "k_deg_pp_erk": 0.0018,# /min
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    4500.0,  # Stim
    160.0,   # Raf_inact
    6.0,     # Raf_act
    7000.0,  # MEK
    300.0,   # MEKp
    9000.0,  # ERK
    250.0,   # ERKp
    85.0,    # PP_MEK
    75.0,    # PP_ERK
]

TSPAN = (0.0, 1500.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Stim, Raf_i, Raf_a, MEK, MEKp, ERK, ERKp, PPm, PPe = Y

    # Stimulus turnover
    v_stim_in = p["k_stim_in"]
    v_stim_deg = p["k_stim_deg"] * Stim

    # Raf activation/inactivation (stimulus-driven)
    v_raf_act = p["k_raf_act"] * Raf_i * Stim
    v_raf_deact = p["k_raf_deact"] * Raf_a

    # MEK phosphorylation by Raf_act (MM in MEK)
    v_mek_phos = p["kcat_mek"] * Raf_a * (MEK / (p["Km_mek"] + MEK))

    # MEK dephosphorylation by PP_MEK (MM in MEKp)
    v_mek_dephos = p["kcat_pp_mek"] * PPm * (MEKp / (p["Km_pp_mek"] + MEKp))

    # ERK phosphorylation by MEKp (MM in ERK; MEKp serves as catalyst)
    v_erk_phos = p["kcat_erk"] * MEKp * (ERK / (p["Km_erk"] + ERK))

    # ERK dephosphorylation by PP_ERK (MM in ERKp)
    v_erk_dephos = p["kcat_pp_erk"] * PPe * (ERKp / (p["Km_pp_erk"] + ERKp))

    # Turnover (MEK/ERK synthesized into unphosphorylated pools)
    v_syn_mek = p["k_syn_mek"]
    v_deg_mek_u = p["k_deg_mek"] * MEK
    v_deg_mek_p = p["k_deg_mek"] * MEKp

    v_syn_erk = p["k_syn_erk"]
    v_deg_erk_u = p["k_deg_erk"] * ERK
    v_deg_erk_p = p["k_deg_erk"] * ERKp

    # Enzyme turnover
    v_syn_raf = p["k_syn_raf"]
    v_deg_raf_i = p["k_deg_raf"] * Raf_i
    v_deg_raf_a = p["k_deg_raf"] * Raf_a

    v_syn_ppm = p["k_syn_pp_mek"]
    v_deg_ppm = p["k_deg_pp_mek"] * PPm

    v_syn_ppe = p["k_syn_pp_erk"]
    v_deg_ppe = p["k_deg_pp_erk"] * PPe

    dStim = v_stim_in - v_stim_deg

    dRaf_i = v_syn_raf - v_deg_raf_i - v_raf_act + v_raf_deact
    dRaf_a = -v_deg_raf_a + v_raf_act - v_raf_deact

    dMEK = v_syn_mek - v_deg_mek_u - v_mek_phos + v_mek_dephos
    dMEKp = -v_deg_mek_p + v_mek_phos - v_mek_dephos

    dERK = v_syn_erk - v_deg_erk_u - v_erk_phos + v_erk_dephos
    dERKp = -v_deg_erk_p + v_erk_phos - v_erk_dephos

    dPPm = v_syn_ppm - v_deg_ppm
    dPPe = v_syn_ppe - v_deg_ppe

    return np.array([dStim, dRaf_i, dRaf_a, dMEK, dMEKp, dERK, dERKp, dPPm, dPPe], dtype=float)

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
    plt.title("signal05_mapk_three_layer (Raf→MEK→ERK, simplified MM cascade)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal05_mapk_three_layer_preview.png", dpi=220)


var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
