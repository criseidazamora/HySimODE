# ============================================================================
# Family 6 (Network Motifs / Multiscale) — Hybrid stochastic–deterministic gene circuit
# motif04_stochastic_gene_deterministic_protein.py
# Hybrid model: stochastic gene expression represented by a promoter switching
# process and mRNA mean–variance dynamics controlling deterministic protein
# production. Protein can be sequestered by an abundant modulator and drives
# catalytic conversion of an abundant substrate pool into a product output.
# Multiscale structure: fast transcriptional dynamics, slower protein turnover,
# and large metabolic pools.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "motif04_stochastic_gene_deterministic_protein"

SPECIES_NAMES = [
    "G_active",      # 0 promoter active state probability (0–1)
    "mRNA_mean",     # 1 mean mRNA copy number (low copy)
    "mRNA_var",      # 2 mRNA variance (noise moment)
    "Protein",       # 3 deterministic protein abundance
    "C_complex",     # 4 protein–modulator complex
    "M_modulator",   # 5 abundant modulator pool (high-copy)
    "S_substrate",   # 6 abundant substrate pool (high-copy)
    "P_product"      # 7 product pool shaped by protein activity (can pulse high)
]

PARAMS = {
    # Promoter switching (two-state gene)
    "k_on": 0.07,        # activation
    "k_off": 0.16,       # inactivation

    # Transcription from active gene
    "k_tx": 11.0,        # mRNA production rate
    "d_m": 2.9,          # mRNA degradation
    "F_tx": 4.8,         # burst Fano factor for mRNA (variance amplification)

    # Translation → deterministic protein
    "k_tl": 9.5,
    "d_p": 0.38,

    # Modulator dynamics and complex formation
    "M_prod": 12.0,      # modulator production rate
    "d_M": 0.020,        # slow modulator turnover
    "k_bind": 0.010,
    "k_unbind": 0.45,
    "d_C": 0.10,

    # Substrate/product metabolic module with an abundant substrate regime
    "S_in": 40.0,        # substrate inflow
    "d_S": 0.010,
    "kcat": 0.11,        # catalytic turnover
    "K_S": 170.0,
    "d_P": 0.28
}

Y0 = np.array([
    0.12,    # G_active
    1.2,     # mRNA_mean
    4.0,     # mRNA_var
    10.0,    # Protein
    0.0,     # C_complex
    320.0,   # M_modulator (abundant -> intended label=0)
    520.0,   # S_substrate (abundant -> intended label=0)
    15.0     # P_product
], dtype=float)

TSPAN = (0.0, 90.0)

# -----------------------------------------------------
# ODE system
# -----------------------------------------------------
def dYdt(t, Y):
    p = PARAMS

    G  = np.clip(Y[0], 0.0, 1.0)
    m  = max(Y[1], 0.0)
    V  = max(Y[2], 0.0)
    Pr = max(Y[3], 0.0)
    C  = max(Y[4], 0.0)
    M  = max(Y[5], 0.0)
    S  = max(Y[6], 0.0)
    P  = max(Y[7], 0.0)

    # Promoter activation probability
    dG = p["k_on"] * (1.0 - G) - p["k_off"] * G

    # mRNA production from active gene + degradation
    dm = p["k_tx"] * G - p["d_m"] * m

    # mRNA variance (bursting approximation; keeps V ~ O(m))
    dV = p["F_tx"] * (p["k_tx"] * G) + p["d_m"] * m - 2.0 * p["d_m"] * V

    # Protein production controlled by mRNA_mean + sequestration by modulator
    v_bind = p["k_bind"] * Pr * M
    v_unbind = p["k_unbind"] * C
    dPr = p["k_tl"] * m - p["d_p"] * Pr - v_bind + v_unbind

    # Complex formation
    dC = v_bind - v_unbind - p["d_C"] * C

    # Modulator dynamics (abundant, slow)
    dM = p["M_prod"] - p["d_M"] * M - v_bind + v_unbind

    # Catalytic conversion S → P driven by free protein
    sat_S = S / (p["K_S"] + S + 1e-12)
    v_cat = p["kcat"] * Pr * sat_S * S

    dS = p["S_in"] - p["d_S"] * S - v_cat
    dP = v_cat - p["d_P"] * P

    return np.array([dG, dm, dV, dPr, dC, dM, dS, dP], dtype=float)

# -----------------------------------------------------
# Main simulation and micro-audit
# -----------------------------------------------------
if __name__ == "__main__":
    t0, tf = TSPAN
    t_span = (t0, tf)
    t_eval = np.linspace(t0, tf, 2000)

    sol = solve_ivp(dYdt, t_span, Y0, t_eval=t_eval, method="Radau")

    if not sol.success:
        print("WARNING: solve_ivp reported failure:", sol.message)

    Y = sol.y

    # Micro-audit 
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        traj = Y[i, :]
        q80 = float(np.quantile(traj, 0.80))
        q99 = float(np.quantile(traj, 0.99))
        label = 1 if (q80 < 200.0 and q99 < 200.0) else 0
        labels.append(label)
        print(f"{name}: q80={q80:.3f}, q99={q99:.3f} => label={label}")

    n1 = int(sum(labels))
    n0 = int(len(labels) - n1)
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass labels detected")

    plt.figure(figsize=(10, 6))
    for i, nm in enumerate(SPECIES_NAMES):
        plt.plot(sol.t, Y[i, :], label=nm)
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title(f"{MODEL_NAME} (hybrid stochastic-gene / deterministic-protein)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_preview.png", dpi=160)
    plt.close()

# -----------------------------------------------------
# Compatibility layer for audit_models.py
# -----------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
