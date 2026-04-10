# ============================================================================
# Family 7 (Stochastic Oscillators) — Activator–inhibitor oscillator with pulsed input
# stochosc06_activator_inhibitor_pulsed_input.py
# Activator–inhibitor oscillator driven by a time-dependent pulsed input.
#
# Deterministic ODE model in which A is a fast activator with cooperative
# self-activation and I is a slower inhibitor induced by A. Activator dynamics
# are further limited by an inhibitor-mediated removal term.
#
# A periodic pulsed input u(t) transiently boosts activator synthesis and
# produces entrained oscillations with sharp transitions and phase-resetting
# behavior. Additional variables represent nutrient-pool and product dynamics.
# ============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

MODEL_NAME = "stochosc06_activator_inhibitor_pulsed_input"

SPECIES_NAMES = [
    "A_activator",
    "I_inhibitor",
    "N_pool",
    "P_product",
]

# -----------------------------
# Parameters
# -----------------------------
PARAMS = {
    # --- Activator synthesis & feedback
    "kA_basal": 0.6,        # basal A synthesis (scaled by nutrient)
    "kA_pulse": 3.5,        # pulse-driven synthesis gain (scaled by nutrient)
    "kA_auto": 6.0,         # autoactivation gain (scaled by nutrient)
    "KA_auto": 120.0,       # autoactivation half-sat
    "n_auto": 3.0,          # cooperativity

    # --- Inhibitor induction
    "kI_prod": 3.2,         # I production gain
    "KI": 140.0,            # half-sat of A->I induction
    "n_I": 2.0,             # cooperativity
    "kI_deg": 0.12,         # I degradation (slow)

    # --- Activator decay & inhibitor-mediated removal (key for oscillations)
    "kA_deg": 0.18,         # linear A degradation
    "kA_inhib": 0.045,      # inhibitor-mediated removal strength
    "K_inhib": 80.0,        # saturation in inhibitor-mediated removal

    # --- Nutrient pool (high-copy context; weakly perturbed by oscillations)
    "kN_in": 18.0,          # nutrient inflow
    "kN_out": 0.03,         # nutrient dilution / turnover
    "kN_cons": 0.09,        # A-dependent nutrient consumption
    "KN_cons": 180.0,       # half-sat for A-dependent consumption

    # --- Product accumulation (tracks A activity; provides additional readout)
    "kP_prod": 1.2,         # product formation gain
    "KP": 120.0,            # half-sat for product formation
    "kP_deg": 0.06,         # product decay / export

    # --- Pulse train definition (Gaussian pulses)
    "pulse_period": 120.0,  # minutes (or arbitrary time units)
    "pulse_sigma": 6.0,     # pulse width (std dev)
    "pulse_dose": 60.0,     # area under each pulse (dose normalization)
    "pulse_t0": 0.0,        # first pulse start reference
    "u_baseline": 0.02,     # small baseline input (keeps system excitable)

    # --- Numerical safety
    "min_state": 0.0,
}

# Initial conditions are set away from steady state to expose transient and oscillatory dynamics.
Y0 = [
    110.0,   # A_activator
    240.0,   # I_inhibitor
    780.0,   # N_pool
    10.0,    # P_product
]

TSPAN = (0.0, 600.0)


# -----------------------------
# Pulsed input u(t)
# -----------------------------
def pulsed_input(t: float, p: dict) -> float:
    """
    Dose-normalized Gaussian pulse train:
      u(t) = u_baseline + sum_k  dose * N( t | t_k, sigma )
    where integral of each Gaussian = dose.
    """
    period = p["pulse_period"]
    sigma = p["pulse_sigma"]
    dose = p["pulse_dose"]
    t0 = p["pulse_t0"]
    u0 = p["u_baseline"]

    # Pulse contributions are evaluated in a local time window because the Gaussian
    # contribution is negligible beyond approximately 5 sigma.
    if sigma <= 0:
        return u0

    # Find nearest pulse index
    k_center = int(np.round((t - t0) / period))
    # Check a few neighbors
    ks = range(k_center - 3, k_center + 4)

    # Amplitude so that integral per pulse is "dose"
    amp = dose / (sigma * np.sqrt(2.0 * np.pi))

    u = u0
    for k in ks:
        tk = t0 + k * period
        z = (t - tk) / sigma
        u += amp * np.exp(-0.5 * z * z)

    return float(u)


# -----------------------------
# ODE system
# -----------------------------
def dYdt(t, y):
    p = PARAMS

    A, I, N, P = y
    A = max(p["min_state"], A)
    I = max(p["min_state"], I)
    N = max(p["min_state"], N)
    P = max(p["min_state"], P)

    u = pulsed_input(t, p)

    # Nutrient scaling (keeps A production linked to resource availability)
    # Saturating dependence prevents runaway.
    N_scale = N / (200.0 + N)

    # Cooperative autoactivation of A
    auto = (A ** p["n_auto"]) / (p["KA_auto"] ** p["n_auto"] + A ** p["n_auto"])

    # A synthesis: basal + pulse-driven + autoactivation, all resource-scaled
    vA_prod = N_scale * (p["kA_basal"] + p["kA_pulse"] * u + p["kA_auto"] * auto)

    # Inhibitor induction by A
    act_to_I = (A ** p["n_I"]) / (p["KI"] ** p["n_I"] + A ** p["n_I"])
    vI_prod = p["kI_prod"] * act_to_I

    # Inhibitor-mediated A removal (saturating in A; proportional to I)
    vA_inhib = p["kA_inhib"] * I * (A / (p["K_inhib"] + A))

    # ODEs
    dA = vA_prod - p["kA_deg"] * A - vA_inhib
    dI = vI_prod - p["kI_deg"] * I

    # Nutrient pool dynamics: inflow - outflow - A-dependent consumption
    vN_cons = p["kN_cons"] * (A / (p["KN_cons"] + A)) * N
    dN = p["kN_in"] - p["kN_out"] * N - vN_cons

    # Product formation follows A activity; decays/exported slowly
    vP_prod = p["kP_prod"] * (A / (p["KP"] + A))
    dP = vP_prod - p["kP_deg"] * P

    return np.array([dA, dI, dN, dP], dtype=float)


if __name__ == "__main__":
    t0, tf = TSPAN
    t_eval = np.linspace(t0, tf, 3000)

    sol = solve_ivp(
        dYdt,
        (t0, tf),
        Y0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-6,
        atol=1e-9,
    )

    # Micro-audit: per-species q80/q99 labels
    labels = []
    for i, name in enumerate(SPECIES_NAMES):
        y = sol.y[i]
        q80 = np.quantile(y, 0.80)
        q99 = np.quantile(y, 0.99)
        label = int((q80 < 200) and (q99 < 200))
        labels.append(label)
        print(f"{name}: q80={q80:.3f}, q99={q99:.3f} => label={label}")

    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"Summary: label1={n1}, label0={n0}")
    if n1 == 0 or n0 == 0:
        print("WARNING: monoclass model")

    # Preview plot of species trajectories
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)

    # Top: regulatory oscillator
    axes[0].plot(sol.t, sol.y[0], label="A_activator")
    axes[0].plot(sol.t, sol.y[1], label="I_inhibitor")
    axes[0].set_ylabel("abundance / concentration")
    axes[0].set_title(f"{MODEL_NAME} (deterministic) — pulsed entrainment")
    axes[0].legend(loc="best")

    # Bottom: metabolic context
    axes[1].plot(sol.t, sol.y[2], label="N_pool")
    axes[1].plot(sol.t, sol.y[3], label="P_product")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("abundance / concentration")
    axes[1].legend(loc="best")

    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_preview.png", dpi=220)

# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y)
