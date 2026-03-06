# stochosc05_brusselator_saturable_deg.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ======================================================================
# Family 7 — Noise-Prone Oscillatory Models
# Model 39: stochosc05_brusselator_saturable_deg.py
#
# Brusselator with biochemically-plausible saturable degradation of X:
# canonical -X is replaced by - Vmax*X/(Kdeg + X).
#
# Key idea: keep the canonical Brusselator structure and ONLY replace
# the linear decay channel. Then tune (Vmax,Kdeg,B) to preserve limit-cycle.
# ======================================================================

MODEL_NAME = "stochosc05_brusselator_saturable_deg"

SPECIES_NAMES = ["X", "Y"]

# --- Default parameters (typically oscillatory) ---
""" PARAMS = {
    "A": 1.0,
    "B": 3.0,      # in canonical model: Hopf if B > 1 + A^2 (i.e. >2 for A=1)
    # Saturable degradation for X: v = Vmax * X/(Kdeg + X)
    # Choose values so v ~ X around X~1 but saturates for larger X.
    "Vmax": 2.2,
    "Kdeg": 1.2,
} """

PARAMS = {
    "A": 1.0,
    "B": 3.0,
    "Vmax": 1.2,
    "Kdeg": 0.2
}


# ICs away from the fixed point
Y0 = [0.20, 5.00]

# Longer horizon helps confirm sustained oscillations (after transients)
TSPAN = (0.0, 200.0)


def v_saturable_deg(X, Vmax, Kdeg):
    return (Vmax * X) / (Kdeg + X + 1e-12)


def dYdt(t, Y, p):
    X, Yc = Y
    A = p["A"]
    B = p["B"]
    Vmax = p["Vmax"]
    Kdeg = p["Kdeg"]

    # Canonical Brusselator term -(B+1)X decomposes as -B*X - X.
    # Replace ONLY the linear decay "-X" with saturable degradation:
    #   -X  ->  -Vmax*X/(Kdeg + X)
    vdeg = v_saturable_deg(X, Vmax, Kdeg)

    dX = A - (B * X) - vdeg + (X * X) * Yc
    dYc = (B * X) - (X * X) * Yc
    return np.array([dX, dYc], dtype=float)


def simulate(p, y0=Y0, tspan=TSPAN, n=3000):
    t_eval = np.linspace(tspan[0], tspan[1], n)
    sol = solve_ivp(
        lambda t, y: dYdt(t, y, p),
        tspan,
        y0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
    )
    return sol


def looks_oscillatory(sol, frac_tail=0.40, amp_min=0.15):
    """
    Deterministic heuristic:
    - take the last frac_tail of the trajectory
    - require non-trivial peak-to-peak amplitude in X and Y
    """
    if (sol is None) or (not sol.success) or sol.y.shape[1] < 10:
        return False

    n = sol.y.shape[1]
    i0 = int((1.0 - frac_tail) * n)

    X = sol.y[0, i0:]
    Y = sol.y[1, i0:]

    ampX = float(np.max(X) - np.min(X))
    ampY = float(np.max(Y) - np.min(Y))

    # Require both to move (Y may be smaller, but not flat)
    return (ampX > amp_min) and (ampY > 0.05 * amp_min)


def autotune_params(base):
    """
    Small deterministic grid search for a limit cycle.
    This is cheap and prevents wasted iterations.
    """
    A = base["A"]

    # Keep A fixed, scan B (drive), and degrad. parameters
    B_grid = [2.4, 2.8, 3.0, 3.4, 4.0, 5.0]
    V_grid = [1.6, 2.0, 2.4, 2.8, 3.2]
    K_grid = [0.6, 1.0, 1.4, 2.0]

    for B in B_grid:
        # In canonical model Hopf threshold is 1 + A^2. Here it's shifted,
        # so we scan above that region.
        if B <= 1.0 + A * A:
            continue
        for Vmax in V_grid:
            for Kdeg in K_grid:
                p = dict(base)
                p["B"] = float(B)
                p["Vmax"] = float(Vmax)
                p["Kdeg"] = float(Kdeg)

                sol = simulate(p, n=2200)
                if looks_oscillatory(sol):
                    return p, sol

    return None, None


def micro_audit(sol):
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
    return labels


if __name__ == "__main__":
    # 1) Try defaults
    sol = simulate(PARAMS)

    # 2) If not oscillatory, autotune deterministically
    if not looks_oscillatory(sol):
        print("Default parameters did not yield a clear limit cycle. Autotuning...")
        tuned_params, tuned_sol = autotune_params(PARAMS)
        if tuned_params is None:
            raise RuntimeError("Autotune failed to find an oscillatory regime.")
        PARAMS = tuned_params
        sol = tuned_sol
        print("Autotune selected parameters:", PARAMS)
    else:
        print("Default parameters produce a limit cycle:", PARAMS)

    # ============================
    # Micro-audit
    # ============================
    micro_audit(sol)

    # ============================
    # Plot
    # ============================
    plt.figure(figsize=(8.5, 5))
    for i in range(sol.y.shape[0]):
        plt.plot(sol.t, sol.y[i], label=SPECIES_NAMES[i])

    plt.xlabel("t")
    plt.ylabel("abundance / concentration")
    plt.title(MODEL_NAME)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODEL_NAME}_preview.png", dpi=220)


# ------------------------------------------------------------
# Compatibility layer for audit_models.py
# ------------------------------------------------------------
var_names = SPECIES_NAMES

def model_odes(t, y):
    return dYdt(t, y, PARAMS)
