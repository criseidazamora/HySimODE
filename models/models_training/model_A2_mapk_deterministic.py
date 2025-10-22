
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===== PARAMETERS (MAPK deterministic high-activity) =========================
params = {
    # Total pools (molecules) -- keep large to enforce high-abundance regime
    "Raf_tot": 15000.0,
    "MEK_tot": 30000.0,
    "ERK_tot": 50000.0,

    # Activation / deactivation rates
    "Stim": 1.0,     # constant upstream stimulus
    "k1":  0.02,     # Raf activation by Stim
    "k2":  0.01,     # Raf* deactivation

    # MEK phosphorylation (two-step) catalyzed by Raf*
    "k3":  2e-5,     # MEK -> MEK-P by Raf*
    "k4":  0.005,    # MEK-P -> MEK (dephos)
    "k5":  2e-5,     # MEK-P -> MEK-PP by Raf*
    "k6":  0.003,    # MEK-PP -> MEK-P (dephos)

    # ERK phosphorylation (two-step) catalyzed by MEK-PP
    "k7":  1.5e-5,   # ERK -> ERK-P by MEK-PP
    "k8":  0.004,    # ERK-P -> ERK (dephos)
    "k9":  1.2e-5,   # ERK-P -> ERK-PP by MEK-PP
    "k10": 0.003,    # ERK-PP -> ERK-P (dephos)
}

# ===== INITIAL CONDITIONS ====================================================
# State vector Y = [Raf*, MEK-P, MEK-PP, ERK-P, ERK-PP]
# Inactive forms are computed by conservation: X = X_tot - sum(active forms)
Y0 = [
    2000.0,   # Raf*
    5000.0,   # MEK-P
    6000.0,   # MEK-PP
    8000.0,   # ERK-P
    9000.0,   # ERK-PP
]

def a2_odes(t, Y):
    p = params
    Raf_star, MEK_p, MEK_pp, ERK_p, ERK_pp = Y

    # Conservation relations
    Raf = p["Raf_tot"] - Raf_star
    MEK = p["MEK_tot"] - MEK_p - MEK_pp
    ERK = p["ERK_tot"] - ERK_p - ERK_pp

    # Raf dynamics
    dRaf_star = p["k1"] * p["Stim"] * Raf - p["k2"] * Raf_star

    # MEK: two-step phos catalyzed by Raf*
    v_mek1 = p["k3"] * Raf_star * MEK          # MEK -> MEK-P
    v_mek2 = p["k5"] * Raf_star * MEK_p        # MEK-P -> MEK-PP
    v_mkdp1 = p["k4"] * MEK_p                  # MEK-P -> MEK
    v_mkdp2 = p["k6"] * MEK_pp                 # MEK-PP -> MEK-P

    dMEK_p  = v_mek1 - v_mkdp1 - v_mek2 + v_mkdp2
    dMEK_pp = v_mek2 - v_mkdp2

    # ERK: two-step phos catalyzed by MEK-PP
    v_erk1 = p["k7"] * MEK_pp * ERK            # ERK -> ERK-P
    v_erk2 = p["k9"] * MEK_pp * ERK_p          # ERK-P -> ERK-PP
    v_ekdp1 = p["k8"] * ERK_p                  # ERK-P -> ERK
    v_ekdp2 = p["k10"] * ERK_pp                # ERK-PP -> ERK-P

    dERK_p  = v_erk1 - v_ekdp1 - v_erk2 + v_ekdp2
    dERK_pp = v_erk2 - v_ekdp2

    return [dRaf_star, dMEK_p, dMEK_pp, dERK_p, dERK_pp]

# ===== SELF-TEST =============================================================
if __name__ == "__main__":
    t_span = (0, 2000)
    t_eval = np.linspace(t_span[0], t_span[1], 1200)
    sol = solve_ivp(a2_odes, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    final = sol.y[:, -1]
    names = ["Raf*", "MEK-P", "MEK-PP", "ERK-P", "ERK-PP"]
    print("[A2] Final values (expect all >> 200):")
    for n, v in zip(names, final):
        print(f"  {n}: {v:.1f}")
    lt200 = (final < 200).sum()
    print(f"[A2] Count < 200: {lt200} (expect 0)")

    import matplotlib.pyplot as plt
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=names[i])
    plt.xlabel("time")
    plt.ylabel("molecules")
    plt.title("A2 MAPK high-activity (deterministic regime)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("A2_preview.png", dpi=200)
