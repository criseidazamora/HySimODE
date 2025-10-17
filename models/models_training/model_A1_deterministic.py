
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===== PARAMETERS (Deterministic high-abundance regime) =====================
# Linear(ish) production-degradation with mild coupling to keep it non-trivial.
# Steady-states ~ alpha / beta, chosen >> 200.
params = {
    "n_species": 12,
    "alpha": [5000, 7000, 8000, 6000, 5500, 9000, 6500, 7200, 6800, 7600, 8100, 8400],
    "beta":  [0.002, 0.0018, 0.0022, 0.0015, 0.002, 0.0016, 0.0021, 0.0017, 0.0019, 0.002, 0.0018, 0.0016],
    "coupling": 0.00005,   # mild global negative feedback to prevent runaway
}

# ===== INITIAL CONDITIONS ====================================================
# Start already high to avoid long transients.
Y0 = [2000.0, 3000.0, 2500.0, 2800.0, 3500.0, 4000.0, 3200.0, 3600.0, 3300.0, 2900.0, 3100.0, 3400.0]

def a1_odes(t, Y):
    p = params
    Y = np.asarray(Y, dtype=float)
    total = np.sum(Y)
    dY = np.zeros_like(Y)
    for i in range(p["n_species"]):
        # dy = alpha_i - beta_i * y_i - coupling * sum(Y)
        dY[i] = p["alpha"][i] - p["beta"][i] * Y[i] - p["coupling"] * total
    return dY

# ===== SELF-TEST (can be commented) =========================================
if __name__ == "__main__":
    t_span = (0, 2000)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(a1_odes, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    final_vals = sol.y[:, -1]
    print("[A1] Final values (should be >> 200):")
    for i, v in enumerate(final_vals):
        print(f"  y{i}: {v:.1f}")
    low = (final_vals < 200).sum()
    print(f"[A1] Count < 200: {low} (expect 0)")

    # Optional plot
    plt.plot(sol.t, sol.y.T)
    plt.xlabel("time")
    plt.ylabel("abundance")
    plt.title("A1 deterministic high-abundance")
    plt.tight_layout()
    plt.savefig("A1_preview.png", dpi=200)
