
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ================= PARAMETERS (A4: Lotka–Volterra) =========================
params = {
    "alpha": 0.025,   # prey intrinsic growth (/min)
    "beta":  1e-6,    # predation rate (1/(molecule*min))
    "delta": 1e-7,    # conversion prey->predator (1/molecule)
    "gamma": 0.02     # predator death (/min)
}

# ================= INITIAL CONDITIONS ======================================
# Y = [X (prey), Y (predator)]
Y0 = [2.0e5, 8.0e4]  # high-copy regime >> 200

def a4_lv_odes(t, Y):
    p = params
    X, Z = Y  # Z = predator
    dX = p["alpha"]*X - p["beta"]*X*Z
    dZ = p["delta"]*X*Z - p["gamma"]*Z
    return [dX, dZ]

# ================= SELF-TEST / PREVIEW =====================================
if __name__ == "__main__":
    t_span = (0, 4000)  # minutes
    t_eval = np.linspace(t_span[0], t_span[1], 1200)
    sol = solve_ivp(a4_lv_odes, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    names = ["Prey X","Predator Y"]
    final = sol.y[:, -1]
    print("[A4] Final values (expect >> 200):")
    for n, v in zip(names, final):
        print(f"  {n}: {v:.1f}")
    lt200 = (final < 200).sum()
    print(f"[A4] Count < 200: {lt200}")

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(sol.t, sol.y[0], label="Prey (X)")
    plt.plot(sol.t, sol.y[1], label="Predator (Y)")
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("A4 Lotka–Volterra (deterministic oscillator, high abundance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("A4_preview.png", dpi=220)
    plt.close()

    # Phase portrait
    plt.figure(figsize=(4.5,4.5))
    plt.plot(sol.y[0], sol.y[1])
    plt.xlabel("Prey X")
    plt.ylabel("Predator Y")
    plt.title("A4 Phase Portrait")
    plt.tight_layout()
    plt.savefig("A4_phase.png", dpi=220)
    plt.close()
