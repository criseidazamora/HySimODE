
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

params = {
    "Glc_ext": 5.0e6,
    "V0":  2.0e4,  "Km0":  2.0e5,
    "V1":  3.0e4,  "Km1":  1.5e5,
    "V2":  3.2e4,  "Km2":  1.0e5,
    "V3":  4.0e4,  "Km3":  8.0e4,
    "V4":  3.5e4,  "Km4":  8.0e4,
    "V5":  5.0e4,  "Km5":  8.0e4,
    "K_ATP_sub": 1.0e5,
    "Ki_ATP":    5.0e5,
    "n_ATP":     2.0,
    "ATP_tot":   5.0e6,
    "k_dil": 1.0e-3
}

Y0 = [2.0e6, 5.0e5, 2.5e5, 1.0e5, 1.5e5, 5.0e5, 3.0e6]

def mm(vmax, Km, S):
    return vmax * S / (Km + S + 1e-12)

def a3_odes(t, Y):
    p = params
    Glc, G6P, F6P, FBP, G3P, Pyr, ATP = Y
    ADP = p["ATP_tot"] - ATP

    v0 = mm(p["V0"], p["Km0"], p["Glc_ext"])
    v1 = mm(p["V1"], p["Km1"], Glc) * (ATP / (p["K_ATP_sub"] + ATP))
    v2 = mm(p["V2"], p["Km2"], G6P)
    factor_s = ATP / (p["K_ATP_sub"] + ATP)
    factor_i = 1.0 / (1.0 + (ATP / p["Ki_ATP"])**p["n_ATP"])
    v3 = mm(p["V3"], p["Km3"], F6P) * factor_s * factor_i
    v4 = mm(p["V4"], p["Km4"], FBP)
    v5 = mm(p["V5"], p["Km5"], G3P)

    dGlc = v0 - v1 - p["k_dil"]*Glc
    dG6P = v1 - v2 - p["k_dil"]*G6P
    dF6P = v2 - v3 - p["k_dil"]*F6P
    dFBP = v3 - v4 - p["k_dil"]*FBP
    dG3P = 2.0*v4 - v5 - p["k_dil"]*G3P
    dPyr = v5 - p["k_dil"]*Pyr
    dATP = -v1 - v3 + 2.0*v5

    return [dGlc, dG6P, dF6P, dFBP, dG3P, dPyr, dATP]

if __name__ == "__main__":
    t_span = (0, 2000)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(a3_odes, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    names = ["Glc","G6P","F6P","FBP","G3P","Pyr","ATP"]
    final = sol.y[:, -1]
    print("[A3] Final values (expect >> 200):")
    for n, v in zip(names, final):
        print(f"  {n}: {v:.1f}")
    lt200 = (final < 200).sum()
    print(f"[A3] Count < 200: {lt200}")

    import matplotlib.pyplot as plt
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=names[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("A3 Reduced Glycolysis with PFK inhibition (deterministic)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("A3_preview.png", dpi=220)
