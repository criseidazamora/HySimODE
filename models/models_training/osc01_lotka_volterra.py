# osc01_lotka_volterra.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================================
# Family 4 (Deterministic Oscillators) — Classical predator–prey (Lotka–Volterra)
# Tuned to show clear oscillations within ~500–2000 min
# ============================================================================
MODEL_NAME = "osc01_lotka_volterra"

SPECIES_NAMES = [
    "Prey",      # S0: prey population / resource (high copy)
    "Pred",      # S1: predator population / consumer (high copy)
    "Nutrient",  # S2: external nutrient pool (high copy; quasi-buffered)
    "Waste",     # S3: waste/byproduct pool (high copy)
    "Ctrl",      # S4: low-copy controller modulating predator efficiency (low copy)
    "Damage",    # S5: low-copy stress/damage in predator (low copy)
]

# ===== PARAMETERS ============================================================
# Key tuning choices for oscillations:
#  - Keep the core LV loop dominant (Prey growth ~ exponential; predation ~ bilinear).
#  - Avoid excessive sinks/damping (moderate mortality; gentle waste clearance).
#  - Make Nutrient a buffered pool (weak uptake, slow decay) so it does not quench cycles.
PARAMS = {
    # Nutrient supply and decay (buffered / weakly perturbed)
    "J_N": 1200.0,       # nutrient input (molecules/min)
    "k_N_decay": 0.0006, # nutrient loss (/min)
    "k_uptake": 2.0e-7,  # weak uptake 1/(molecule*min) -> keeps N quasi-buffered
    "K_N": 4000.0,       # nutrient half-sat (molecules)

    # Prey growth on nutrient (near-exponential when N buffered)
    "y_growth": 0.055,   # growth gain (/min)

    # Predation and predator reproduction (core LV coupling)
    "k_pred": 4.0e-6,    # predation rate 1/(molecule*min)
    "y_pred": 0.045,     # conversion gain (dimensionless)

    # Natural decay (moderate to preserve oscillations)
    "d_prey": 0.0030,    # prey loss (/min)
    "d_pred": 0.0060,    # predator loss (/min)

    # Waste production and clearance (do NOT over-clear)
    "k_waste_from_prey": 0.0009,  # /min
    "k_waste_from_pred": 0.0012,  # /min
    "k_waste_clear": 0.0012,      # /min

    # Low-copy controller (slow, mild effect)
    "alpha_ctrl": 0.28,   # synthesis (molecules/min)
    "delta_ctrl": 0.010,  # degradation (/min)
    "K_ctrl": 60.0,       # scale (molecules)
    "n_ctrl": 2.0,        # cooperativity

    # Predator damage (kept weak to avoid damping out oscillations)
    "k_dmg_form": 4.0e-6,  # scaled by predation load
    "k_dmg_repair": 0.012, # /min
    "K_dmg": 85.0,         # scale (molecules)
}

# ===== INITIAL CONDITIONS ====================================================
Y0 = [
    7000.0,  # Prey
    2200.0,  # Pred
    12000.0, # Nutrient
    800.0,   # Waste
    35.0,    # Ctrl
    15.0,    # Damage
]

TSPAN = (0.0, 3000.0)  # minutes

def dYdt(t, Y):
    p = PARAMS
    Prey, Pred, N, W, Ctrl, Dmg = Y

    # Nutrient uptake by prey (kept weak -> Nutrient buffered)
    v_uptake = p["k_uptake"] * Prey * N
    sat_N = N / (p["K_N"] + N)

    # Prey growth (approximately exponential when N buffered)
    v_growth = p["y_growth"] * Prey * (0.3 + 0.7 * sat_N)

    # Controller increases predator efficiency (dimensionless multiplier)
    eff_ctrl = 1.0 + (Ctrl ** p["n_ctrl"]) / (p["K_ctrl"] ** p["n_ctrl"] + Ctrl ** p["n_ctrl"])

    # Damage reduces predation efficiency (dimensionless inhibition)
    eff_dmg = 1.0 / (1.0 + (Dmg / p["K_dmg"]))

    # Predation interaction (core LV term)
    v_predation = p["k_pred"] * eff_ctrl * eff_dmg * Prey * Pred

    # Predator growth from predation
    v_pred_growth = p["y_pred"] * v_predation

    # Nutrient dynamics
    dN = p["J_N"] - p["k_N_decay"] * N - v_uptake

    # Prey dynamics
    dPrey = v_growth - v_predation - p["d_prey"] * Prey

    # Predator dynamics
    dPred = v_pred_growth - p["d_pred"] * Pred

    # Waste dynamics (gentle clearance)
    v_waste = p["k_waste_from_prey"] * Prey + p["k_waste_from_pred"] * Pred
    dW = v_waste - p["k_waste_clear"] * W

    # Low-copy controller turnover
    dCtrl = p["alpha_ctrl"] - p["delta_ctrl"] * Ctrl

    # Predator damage accumulates with predation load and is repaired (kept weak)
    v_dmg_form = p["k_dmg_form"] * v_predation
    dDmg = v_dmg_form - p["k_dmg_repair"] * Dmg

    return np.array([dPrey, dPred, dN, dW, dCtrl, dDmg], dtype=float)

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

    # Plot (required for oscillator correction)
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])
    plt.xlabel("time [min]")
    plt.ylabel("molecules")
    plt.title("osc01_lotka_volterra (tuned oscillatory predator–prey)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("osc01_lotka_volterra_preview.png", dpi=220)


var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
