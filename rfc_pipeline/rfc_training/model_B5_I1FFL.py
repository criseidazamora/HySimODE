
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

params = {
    "alpha_mX": 0.5,
    "alpha_mY": 0.45,
    "alpha_mZ": 0.5,
    "beta_p": 5.0,
    "delta_m": 0.2,
    "delta_p": 0.01,
    "K_act_X": 60.0,
    "n_act": 2.0,
    "K_rep_Y": 70.0,
    "n_rep": 2.5,
}

Y0 = [3.0, 10.0, 2.0, 5.0, 2.0, 5.0]

def hill_act(x, K, n):
    return (x**n) / (K**n + x**n + 1e-12)

def hill_rep(y, K, n):
    return 1.0 / (1.0 + (y / K)**n)

def b5_i1ffl_odes(t, Y):
    p = params
    mX, pX, mY, pY, mZ, pZ = Y
    dmX = p["alpha_mX"] - p["delta_m"] * mX
    dmY = p["alpha_mY"] * hill_act(pX, p["K_act_X"], p["n_act"]) - p["delta_m"] * mY
    dmZ = p["alpha_mZ"] * hill_act(pX, p["K_act_X"], p["n_act"]) * hill_rep(pY, p["K_rep_Y"], p["n_rep"]) - p["delta_m"] * mZ
    dpX = p["beta_p"] * mX - p["delta_p"] * pX
    dpY = p["beta_p"] * mY - p["delta_p"] * pY
    dpZ = p["beta_p"] * mZ - p["delta_p"] * pZ
    return [dmX, dpX, dmY, dpY, dmZ, dpZ]

if __name__ == "__main__":
    t_span = (0, 1200)
    t_eval = np.linspace(t_span[0], t_span[1], 800)
    sol = solve_ivp(b5_i1ffl_odes, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)
    names = ["mX","pX","mY","pY","mZ","pZ"]
    final = sol.y[:, -1]
    print("[B5 I1-FFL] Final values (expect low-copy <~200):")
    for n, v in zip(names, final):
        print(f"  {n}: {v:.2f}")
    lt200 = (final < 200).sum()
    print(f"[B5 I1-FFL] Count < 200: {lt200} / 6")
    plt.figure(figsize=(6,4))
    plt.plot(sol.t, sol.y[1], label="pX")
    plt.plot(sol.t, sol.y[3], label="pY")
    plt.plot(sol.t, sol.y[5], label="pZ")
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("B5 I1-FFL (proteins) - low abundance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("B5_preview.png", dpi=220)
    plt.close()
