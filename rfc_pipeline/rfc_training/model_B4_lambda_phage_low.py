"""
model_B4_lambda_phage_low.py
Simplified lambda phage lysis–lysogeny genetic switch.
Low-copy regime for training as purely stochastic (label=1).
"""

import numpy as np

# ========== Parameters ==========
params = {
    # mRNA transcription rates
    "alpha_cI": 0.5,   # basal cI mRNA synthesis
    "alpha_Cro": 0.5,  # basal Cro mRNA synthesis
    # mRNA degradation
    "delta_m": 0.08,
    # Protein translation rates
    "beta_cI": 1.1,
    "beta_Cro": 1.1,
    # Protein degradation
    "delta_p": 0.06,
    # Hill repression parameters
    "K_cI": 6.0,
    "K_Cro": 6.0,
    "n": 2.0
}

"""     "alpha_m_cI": 0.50,
    "alpha_m_Cro": 0.50,
    "beta_p_cI": 1.10,
    "beta_p_Cro": 1.10,
    "delta_m": 0.08,
    "delta_p": 0.06,
    "K_cI_auto": 8.0,
    "K_cI_repress": 6.0,
    "K_Cro_repress": 6.0,
    "n_auto": 3.0,    # entero -> sin problemas de potencias
    "n_rep": 2.0      # entero -> sin problemas de potencias
 """
# ========== Initial conditions ==========
# [m_cI, p_cI, m_Cro, p_Cro]
Y0 = [
    3.0,   # m_cI
    35.0,  # p_cI
    3.0,   # m_Cro
    10.0   # p_Cro
]

# Optional: species names
var_names = ["m_cI", "p_cI", "m_Cro", "p_Cro"]

# ========== ODE system ==========
def hill_repression(x, K, n):
    return 1.0 / (1.0 + (x / K)**n)

def b4_lambda_odes(t, Y):
    m_cI, p_cI, m_Cro, p_Cro = Y
    p = params

    # Transcription with mutual repression
    dm_cI = p["alpha_cI"] * hill_repression(p_Cro, p["K_Cro"], p["n"]) - p["delta_m"] * m_cI
    dm_Cro = p["alpha_Cro"] * hill_repression(p_cI, p["K_cI"], p["n"]) - p["delta_m"] * m_Cro

    # Translation
    dp_cI = p["beta_cI"] * m_cI - p["delta_p"] * p_cI
    dp_Cro = p["beta_Cro"] * m_Cro - p["delta_p"] * p_Cro

    return [dm_cI, dp_cI, dm_Cro, dp_Cro]

# ========== Self-test ==========
if __name__ == "__main__":
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    t_span = (0, 2000)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(b4_lambda_odes, t_span, Y0, t_eval=t_eval, method="Radau", rtol=1e-7, atol=1e-9)

    for name, vals in zip(var_names, sol.y):
        plt.plot(sol.t, vals, label=name)
    plt.xlabel("Time (min)")
    plt.ylabel("Molecules")
    plt.title("B4 lambda phage low-copy dynamics")
    plt.legend()
    plt.tight_layout()
    plt.savefig("B4_lambda_phage_low_preview.png", dpi=200)
  
