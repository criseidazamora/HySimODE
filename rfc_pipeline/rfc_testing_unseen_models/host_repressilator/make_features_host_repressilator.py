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

def import_module(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    spec = util.spec_from_file_location(name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def find_odes_fn(mod) -> Optional[Callable]:
    # If no use of mod.main() or prefer going directly to ODE.
    candidates = ["repressilator_odes", "host_rep_odes", "odes"]
    for fn in candidates:
        if hasattr(mod, fn):
            return getattr(mod, fn)
    for attr in dir(mod):
        if attr.endswith("_odes") and callable(getattr(mod, attr)):
            return getattr(mod, attr)
    return None

def get_names(mod, nvars: int) -> List[str]:
    # if we add var_names in modeule; otherwise Y0..Y{n-1}
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

        # All trajectory
        mean_total = float(np.mean(y))
        std_total = float(np.std(y))
        cv_total = float(std_total / (mean_total + 1e-12))
        madydt_total = float(np.mean(np.abs(dy)))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        minmax_ratio = float((ymin + 1e-12) / (ymax + 1e-12))

        # Final window
        yf = y[mask_final]
        dyf = dy[mask_final]
        mean_final = float(np.mean(yf))
        std_final = float(np.std(yf))
        cv_final = float(std_final / (mean_final + 1e-12))
        madydt_final = float(np.mean(np.abs(dyf)))
        final_value = float(y[-1])

        # Initial window
        yi = y[mask_initial]
        mean_initial = float(np.mean(yi))

        rows.append({
            "mean_total": mean_total,
            "std_total": std_total,
            "cv_total": cv_total,
            "madydt_total": madydt_total,
            "min": ymin, "max": ymax, "minmax_ratio": minmax_ratio,
            "final_mean": mean_final, "final_std": std_final,
            "final_cv": cv_final, "final_madydt": madydt_final,
            "final_value": final_value,
            "initial_mean": mean_initial,
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
        sol = solve_ivp(odes, (0.0, 5000.0), y0, method="BDF", rtol=1e-6, atol=1e-8,
                        t_eval=np.arange(0.0, 5000.0+1.0, 1.0))
        t, Y = sol.t, sol.y

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
