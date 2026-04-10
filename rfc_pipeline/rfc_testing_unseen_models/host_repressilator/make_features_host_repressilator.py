# =============================================================================
#  make_features_host_repressilator.py
# -----------------------------------------------------------------------------
#  Description:
#      This script computes quantitative trajectory features for each species
#      in the host–repressilator hybrid model. The same 13 features used during
#      Random Forest Classifier (RFC) training are extracted here from the
#      simulated ODE trajectories. These features are later used for testing
#      the trained RFC on unseen models.
#
#  Features computed per species:
#      mean_total, std_total, cv_total, madydt_total, min, max, minmax_ratio,
#      final_mean, final_std, final_cv, final_madydt, final_value, initial_mean
#
#  Output:
#      - A CSV file (features_host_repressilator.csv) containing one row per
#        species with all computed features.
#
#  Notes:
#      - The feature definitions, observation windows, and normalization
#        parameters are consistent with the training dataset used in
#        HySimODE’s RFC calibration.
#      - If the model module defines a 'main()' function, it will be used to
#        run the full simulation pipeline; otherwise, the script falls back
#        to direct ODE integration.
#
#  © 2025 Criseida G. Zamora Chimal. 
# =============================================================================

import os
import numpy as np
import pandas as pd
from importlib import util
from typing import Optional, Callable, List

# === Config ===
MODEL_FILE = "host_repressilator.py"   # Change the name of this file if the file changes
OBS_WINDOW = 100.0
INITIAL_WINDOW = 50.0
OUT_CSV = "features_host_repressilator.csv"
EPS = 1e-12
T_FINAL = 1000.0
DT = 1.0

def import_module(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    spec = util.spec_from_file_location(name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def find_odes_fn(mod) -> Optional[Callable]:
    
    candidates = ["repressilator_odes", "host_rep_odes", "odes"]
    for fn in candidates:
        if hasattr(mod, fn):
            return getattr(mod, fn)
    for attr in dir(mod):
        if attr.endswith("_odes") and callable(getattr(mod, attr)):
            return getattr(mod, attr)
    return None

def get_names(mod, nvars: int) -> List[str]:
    
    if hasattr(mod, "var_names"):
        try:
            names = list(getattr(mod, "var_names"))
            if len(names) == nvars:
                return names
        except Exception:
            pass
    return [f"Y{i}" for i in range(nvars)]

def compute_features(t: np.ndarray, Y: np.ndarray):
    nvars = Y.shape[0]
    t_end = t[-1]
    mask_final = t >= (t_end - OBS_WINDOW)
    mask_initial = t <= INITIAL_WINDOW

    rows = []
    for i in range(nvars):
        y = np.maximum(Y[i, :], 0.0)
        dy = np.gradient(y, t)
   
        # ---------------- Whole-trajectory features ----------------
        mean_total = float(np.mean(y))
        std_total = float(np.std(y))
        cv_total = float(std_total / (mean_total + EPS))
        madydt_total = float(np.mean(np.abs(dy)))

        ymin, ymax = float(np.min(y)), float(np.max(y))
        minmax_ratio = float((ymin + EPS) / (ymax + EPS))

        # ---------------- Final-window features --------------------
        yf = y[mask_final]
        yi = y[mask_initial]

        # ---------------- Robust windowed features  ----------
        # q50 (median) provides a robust estimate of typical abundance per window.
        initial_q50 = float(np.quantile(yi, 0.50)) if yi.size else float("nan")
        final_q50 = float(np.quantile(yf, 0.50)) if yf.size else float("nan")

        # CV (dimensionless) captures relative variability in each window.
        initial_std = float(np.std(yi)) if yi.size else float("nan")
        initial_cv = float(initial_std / (float(np.mean(yi)) + EPS)) if yi.size else float("nan")

        final_std = float(np.std(yf)) if yf.size else float("nan")
        final_cv = float(final_std / (float(np.mean(yf)) + EPS)) if yf.size else float("nan")

        # Min/max ratio (dimensionless) captures oscillatory amplitude / transients without scaling.
        if yi.size:
            initial_min = float(np.min(yi))
            initial_max = float(np.max(yi))
            initial_minmax_ratio = float((initial_min + EPS) / (initial_max + EPS))
        else:
            initial_minmax_ratio = float("nan")

        if yf.size:
            final_min = float(np.min(yf))
            final_max = float(np.max(yf))
            final_minmax_ratio = float((final_min + EPS) / (final_max + EPS))
        else:
            final_minmax_ratio = float("nan")

        # Normalized mean absolute derivative (dimensionless) captures relative rate of change.
        if yi.size:
            initial_madydt = float(np.mean(np.abs(dy[mask_initial])))
            initial_nmadydt = float(initial_madydt / (initial_q50 + EPS))
        else:
            initial_nmadydt = float("nan")

        if yf.size:
            final_madydt2 = float(np.mean(np.abs(dy[mask_final])))
            final_nmadydt = float(final_madydt2 / (final_q50 + EPS))
        else:
            final_nmadydt = float("nan")

        # Whole-trajectory normalized MAD (dimensionless)
        median_total = float(np.quantile(y, 0.50))
        nmadydt_total = float(madydt_total / (median_total + EPS))
        minmax_ratio_total = minmax_ratio
        rows.append({
            "initial_q50": initial_q50,
            "initial_cv": initial_cv,
            "initial_minmax_ratio": initial_minmax_ratio,
            "initial_nmadydt": initial_nmadydt,

            "final_q50": final_q50,
            "final_cv": final_cv,
            "final_minmax_ratio": final_minmax_ratio,
            "final_nmadydt": final_nmadydt,

            "cv_total": cv_total,
            "minmax_ratio_total": minmax_ratio_total,
            "nmadydt_total": nmadydt_total,
        })
    return rows

def main():
    mod = import_module(MODEL_FILE)

    # Using main() as makes wrap-ups and integrates the final system 
    t, Y = None, None
    if hasattr(mod, "main") and callable(mod.main):
        # Inactive plots to speed up
        if hasattr(mod, "PLOT"):
            setattr(mod, "PLOT", False)
        # Execute pipeline: return y0 y sol
        y0, sol = mod.main()
        t, Y = sol.t, sol.y
    else:
        # Fallback: if we don't use main(), we integrate directly (mod.Y0 and ODE)
        from scipy.integrate import solve_ivp
        odes = find_odes_fn(mod)
        if odes is None:
            raise AttributeError("ODE function not found (as *_odes), not main().")
        if not hasattr(mod, "Y0"):
            raise AttributeError(" Y0 not found in module and there is not main().")
        y0 = np.array(mod.Y0, dtype=float)
        # We should adjust as required
        from scipy.integrate import solve_ivp

        # --- Align with HySimODE deterministic pre-simulation ---
        solver_options = getattr(mod, "solver_options", {
            "method": "BDF",
            "rtol": 1e-6,
            "atol": 1e-9,
            "max_step": DT
        })
        solver_options = dict(solver_options)
        solver_options["max_step"] = DT  # force exact match

        t_eval = np.linspace(0.0, T_FINAL, int(T_FINAL / DT) + 1)

        sol = solve_ivp(
            lambda t, y: odes(t, y, getattr(mod, "params", {})),
            [0.0, T_FINAL],
            y0,
            t_eval=t_eval,
            **solver_options
        )

        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        t, Y = sol.t, np.maximum(sol.y, 0.0)

    feats = compute_features(t, Y)
    names = get_names(mod, Y.shape[0])
    df = pd.DataFrame(feats)
    df.insert(0, "species_name", names)
    df.insert(0, "model", os.path.splitext(MODEL_FILE)[0])
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved features → {OUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
