
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===== PARAMETERS (Repressilator - low abundance) ============================
params = {
    # Production rates (molecules/min) -- small to keep low counts
    "alpha_m": 0.8,     # mRNA basal synthesis
    "alpha_p": 2.0,     # protein synthesis per mRNA

    # Degradation rates (per min)
    "delta_m": 0.08,    # mRNA decay
    "delta_p": 0.03,    # protein decay

    # Repression
    "K": 50.0,          # dissociation constant (molecules)
    "n": 2.5,           # Hill coefficient

    # Translation delay proxy via limited ribosomes (optional mild saturation)
    "k_trans_sat": 200.0
}

# State: [m1, p1, m2, p2, m3, p3]
Y0 = [5.0, 10.0, 6.0, 8.0, 4.0, 12.0]

def hill_repression(x, K, n):
    return 1.0 / (1.0 + (x / K)**n)

def b2_odes(t, Y):
    p = params
    m1, p1, m2, p2, m3, p3 = Y

    # Transcription with repression by previous protein in the ring
    dm1 = p["alpha_m"] * hill_repression(p3, p["K"], p["n"]) - p["delta_m"] * m1
    dm2 = p["alpha_m"] * hill_repression(p1, p["K"], p["n"]) - p["delta_m"] * m2
    dm3 = p["alpha_m"] * hill_repression(p2, p["K"], p["n"]) - p["delta_m"] * m3

    # Translation with mild saturation proxy + degradation
    trans1 = p["alpha_p"] * m1 / (1.0 + (p1 / p["k_trans_sat"]))
    trans2 = p["alpha_p"] * m2 / (1.0 + (p2 / p["k_trans_sat"]))
    trans3 = p["alpha_p"] * m3 / (1.0 + (p3 / p["k_trans_sat"]))

    dp1 = trans1 - p["delta_p"] * p1
    dp2 = trans2 - p["delta_p"] * p2
    dp3 = trans3 - p["delta_p"] * p3

    return [dm1, dp1, dm2, dp2, dm3, dp3]

# ===== SELF-TEST =============================================================
if __name__ == "__main__":
    t_span = (0, 3000)
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    sol = solve_ivp(b2_odes, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    final = sol.y[:, -1]
    names = ["m1", "p1", "m2", "p2", "m3", "p3"]
    print("[B2] Final values (expect all << 200):")
    for n, v in zip(names, final):
        print(f"  {n}: {v:.2f}")
    lt200 = (final < 200).sum()
    print(f"[B2] Count < 200: {lt200} (expect 6)")

    import matplotlib.pyplot as plt
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=names[i])
    plt.xlabel("time")
    plt.ylabel("molecules")
    plt.title("B2 Repressilator (low-abundance regime)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("B2_preview.png", dpi=220)
