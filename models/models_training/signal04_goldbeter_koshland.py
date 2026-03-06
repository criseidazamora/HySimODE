# signal04_goldbeter_koshland.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 2 (Signaling) — Goldbeter–Koshland ultrasensitive switch
# ============================================================================
MODEL_NAME = "signal04_goldbeter_koshland"

SPECIES_NAMES = [
    "Stim",     # S0: upstream stimulus controlling kinase activation (high-ish copy, decays)
    "K_inact",  # S1: inactive kinase (low copy)
    "K_act",    # S2: active kinase (low copy)
    "Pase",     # S3: phosphatase (low copy)
    "S",        # S4: unmodified substrate (high copy)
    "Sp",       # S5: modified substrate (high copy)
]

# ===== PARAMETERS ============================================================
# Goldbeter–Koshland switch arises when kinase and phosphatase operate near saturation
# on a conserved substrate pool, producing an ultrasensitive steady-state fraction Sp.
PARAMS = {
    # Stimulus turnover (creates early vs late behavior)
    "k_stim_in": 0.0,
    "k_stim_deg": 0.006,   # /min

    # Kinase activation/inactivation driven by stimulus
    "k_act": 2.5e-5,       # K_inact + Stim -> K_act (1/(molecule*min))
    "k_deact": 0.012,      # K_act -> K_inact (/min)

    # Michaelis–Menten modification rates (copy-number form)
    # v_mod = VmaxK * S / (KmK + S), with VmaxK = kcatK * K_act
    "kcatK": 1.8,          # /min
    "KmK": 800.0,          # molecules

    # v_demod = VmaxP * Sp / (KmP + Sp), with VmaxP = kcatP * Pase
    "kcatP": 1.3,          # /min
    "KmP": 700.0,          # molecules

    # Slow protein turnover (keeps biology plausible while preserving near-conservation)
    "k_syn_S": 3.0,        # molecules/min
    "k_deg_S": 2.0e-4,     # /min

    # Enzyme turnover (low-copy)
    "k_syn_K": 0.20,       # molecules/min
    "k_deg_K": 0.0015,     # /min
    "k_syn_P": 0.16,       # molecules/min
    "k_deg_P": 0.0016,     # /min
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    5000.0,  # Stim
    130.0,   # K_inact
    6.0,     # K_act
    90.0,    # Pase
    9000.0,  # S
    500.0,   # Sp
]

TSPAN = (0.0, 1500.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Stim, Ki, Ka, P, S, Sp = Y

    # Stimulus turnover (decay after initial condition)
    v_stim_in = p["k_stim_in"]
    v_stim_deg = p["k_stim_deg"] * Stim

    # Kinase activation/inactivation
    v_K_act = p["k_act"] * Ki * Stim
    v_K_deact = p["k_deact"] * Ka

    # Goldbeter–Koshland core: opposing saturated MM modification/demodification
    VmaxK = p["kcatK"] * Ka
    VmaxP = p["kcatP"] * P

    v_mod = VmaxK * (S / (p["KmK"] + S))
    v_demod = VmaxP * (Sp / (p["KmP"] + Sp))

    # Slow turnover (substrate synthesized into unmodified pool)
    v_syn_S = p["k_syn_S"]
    v_deg_S_u = p["k_deg_S"] * S
    v_deg_S_p = p["k_deg_S"] * Sp

    # Enzyme turnover
    v_syn_K = p["k_syn_K"]
    v_deg_Ki = p["k_deg_K"] * Ki
    v_deg_Ka = p["k_deg_K"] * Ka

    v_syn_P = p["k_syn_P"]
    v_deg_P = p["k_deg_P"] * P

    dStim = v_stim_in - v_stim_deg

    dKi = v_syn_K - v_deg_Ki - v_K_act + v_K_deact
    dKa = -v_deg_Ka + v_K_act - v_K_deact

    dP = v_syn_P - v_deg_P

    dS = v_syn_S - v_deg_S_u - v_mod + v_demod
    dSp = -v_deg_S_p + v_mod - v_demod

    return np.array([dStim, dKi, dKa, dP, dS, dSp], dtype=float)

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
    plt.title("signal04_goldbeter_koshland (ultrasensitive covalent-modification switch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal04_goldbeter_koshland_preview.png", dpi=220)

var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
