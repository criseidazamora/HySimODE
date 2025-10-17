
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# B3_faithful: Toggle switch (Gardner et al., Nature 2000) — faithful structure
# ============================================================================
# Equations (per gene i in {A,B}):
#   dm_i/dt = alpha_m_i / (1 + (p_j/K_i)**n_i) - delta_m * m_i
#   dp_i/dt = beta_p * m_i - delta_p * p_i
# with j≠i and symmetric parameters by default.
#
# Reference:
# Gardner, T. S., Cantor, C. R., & Collins, J. J. (2000).
# Construction of a genetic toggle switch in Escherichia coli.
# Nature, 403(6767), 339–342. doi:10.1038/35002131

params = {
    # Transcription (mRNA/min) — moderate to allow basal expression when repressed
    "alpha_mA": 0.6,
    "alpha_mB": 0.6,

    # Repression thresholds (proteins) — set in the tens–hundreds range
    "K_A": 70.0,     # threshold for repression of A by pB
    "K_B": 70.0,     # threshold for repression of B by pA

    # Hill coefficients — cooperative binding of repressors
    "n_A": 2.0,
    "n_B": 2.0,

    # Translation (proteins per mRNA per min)
    "beta_p": 5.0,

    # Degradation/dilution rates (/min)
    "delta_m": 0.14,  # mRNA half-life ~ 5 min
    "delta_p": 0.006, # protein half-life ~ 115 min
}

# Initial conditions: balanced to avoid trivial dominance
# Y = [mA, pA, mB, pB]
Y0 = [
    5.0,    # mA
    200.0,  # pA
    5.0,    # mB
    200.0   # pB
]

def hill_repression(p, K, n):
    return 1.0 / (1.0 + (p / K)**n)

def b3_faithful_odes(t, Y):
    p = params
    mA, pA, mB, pB = Y

    dmA = p["alpha_mA"] * hill_repression(pB, p["K_A"], p["n_A"]) - p["delta_m"] * mA
    dmB = p["alpha_mB"] * hill_repression(pA, p["K_B"], p["n_B"]) - p["delta_m"] * mB

    dpA = p["beta_p"] * mA - p["delta_p"] * pA
    dpB = p["beta_p"] * mB - p["delta_p"] * pB

    return [dmA, dpA, dmB, dpB]

# ================= SELF-TEST / PREVIEW ======================================
if __name__ == "__main__":
    t_span = (0, 3000)
    t_eval = np.linspace(t_span[0], t_span[1], 1200)
    sol = solve_ivp(b3_faithful_odes, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    names = ["mA", "pA", "mB", "pB"]
    final = sol.y[:, -1]
    print("[B3_faithful] Final values:")
    for n, v in zip(names, final):
        print(f"  {n}: {v:.2f}")
    lt200 = (final < 200).sum()
    print(f"[B3_faithful] Count < 200: {lt200}")

    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=names[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("B3_faithful Toggle Switch (Gardner et al. structure)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("B3_faithful_preview.png", dpi=220)
