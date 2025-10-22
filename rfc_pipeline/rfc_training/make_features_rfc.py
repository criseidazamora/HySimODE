# =============================================================================
#  make_features_rfc.py
# -----------------------------------------------------------------------------
#  Description:
#      This script simulates a collection of deterministic biochemical models
#      and computes quantitative trajectory features for each species.
#      The extracted features are used to train the Random Forest Classifier (RFC)
#      in HySimODE for distinguishing between stochastic and deterministic regimes.
#
#  Features computed per species:
#      mean_total, std_total, cv_total, madydt_total,
#      min, max, minmax_ratio,
#      final_mean, final_std, final_cv, final_madydt,
#      final_value, initial_mean
#
#  Labeling rule:
#      A species is labeled as stochastic (1) if
#      final_mean < 200, mean_total < 200, and initial_mean < 200.
#      Otherwise, it is labeled as deterministic (0).
#
#  Output:
#      - A CSV file (features_rfc.csv) containing one row per species
#        with all computed features and labels.
#
#  Notes:
#      - Each model file must define an ODE function and an initial condition (Y0).
#      - The integration uses solve_ivp with the Radau method for stiff systems.
#      - Observation and initial windows are used to extract temporal statistics.
#
#  © 2025 Criseida G. Zamora Chimal.
# =============================================================================

import os
import numpy as np
import pandas as pd
from importlib import util
from scipy.integrate import solve_ivp
from typing import Callable, Dict, Any, List, Optional

# ---------- Configuration ----------
MODEL_FILES = [
    "model_A1_deterministic.py",
    "model_A2_mapk_deterministic.py",
    "model_A3_glycolysis_reduced.py",
    "model_A4_LotkaVolterra.py",
    "model_B2_repressilator_low.py",
    "model_B3_toggle_switch_faithful.py",
    "model_B4_lambda_phage_low.py",
    "model_B5_I1FFL.py",
    "host_odes.py",
]

T_FINAL = 2000.0
N_POINTS = 1000
OBS_WINDOW = 100.0
INITIAL_WINDOW = 50.0  # Change: initial window used to compute initial_mean
ATOL = 1e-9
RTOL = 1e-7
METHOD = "Radau"

OUT_CSV = "features_rfc.csv"

# ---------- Helpers ----------
def try_import_model(path: str):
    """Attempt to import a Python model module dynamically."""
    if not os.path.exists(path):
        return None
    name = os.path.splitext(os.path.basename(path))[0]
    spec = util.spec_from_file_location(name, path)
    mod = util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    except Exception as e:
        print(f"[WARN] Could not import {path}: {e}")
        return None

def get_ode_fn(mod) -> Optional[Callable]:
    """Find and return the ODE function inside the imported model."""
    candidates = [
        "a1_odes", "a2_odes", "a3_odes", "a4_lv_odes",
        "b2_odes", "b3_faithful_odes", "b4_lambda_odes",
        "b5_i1ffl_odes", "host_odes"
    ]
    for fn in candidates:
        if hasattr(mod, fn):
            return getattr(mod, fn)
    for attr in dir(mod):
        if attr.endswith("_odes") and callable(getattr(mod, attr)):
            return getattr(mod, attr)
    return None

def get_names(mod, nvars: int) -> List[str]:
    """Return variable names if defined in the module; otherwise use generic labels."""
    if hasattr(mod, "var_names"):
        try:
            names = list(getattr(mod, "var_names"))
            if len(names) == nvars:
                return names
        except Exception:
            pass
    return [f"Y{i}" for i in range(nvars)]

def simulate_model(mod, t_final=T_FINAL, n_points=N_POINTS):
    """Integrate the ODE model using solve_ivp."""
    y0 = np.array(mod.Y0, dtype=float)
    odes = get_ode_fn(mod)
    if odes is None:
        raise RuntimeError("No ODE function found in module.")
    t_span = (0.0, t_final)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method=METHOD, rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError("solve_ivp failed")
    return sol.t, sol.y

def compute_features(t: np.ndarray, Y: np.ndarray, obs_window: float):
    """Compute all 13 quantitative features for each species."""
    nt = t.size
    nvars = Y.shape[0]
    t_end = t[-1]
    mask_final = t >= (t_end - obs_window)
    mask_initial = t <= INITIAL_WINDOW  # Change: initial window mask
    feats = []
    for i in range(nvars):
        y = Y[i, :]
        dy = np.gradient(y, t)
        # Whole trajectory
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
        # 
        yi = y[mask_initial]
        mean_initial = float(np.mean(yi))
        feats.append({
            "mean_total": mean_total,
            "std_total": std_total,
            "cv_total": cv_total,
            "madydt_total": madydt_total,
            "min": ymin, "max": ymax, "minmax_ratio": minmax_ratio,
            "final_mean": mean_final, "final_std": std_final,
            "final_cv": cv_final, "final_madydt": madydt_final,
            "final_value": float(y[-1]),
            "initial_mean": mean_initial,  # 
        })
    return feats

def label_rule(feature_row: Dict[str, Any]) -> int:
    """Binary labeling rule: 1 = stochastic, 0 = deterministic."""
    return int(
        (feature_row["final_mean"] < 200.0) and
        (feature_row["mean_total"] < 200.0) and
        (feature_row["initial_mean"] < 200.0)
    )

# ---------- Main run ----------
rows = []
loaded_models = []
for fname in MODEL_FILES:
    mod = try_import_model(fname)
    if mod is None:
        print(f"[SKIP] Model not found or failed to import: {fname}")
        continue
    try:
        t, Y = simulate_model(mod)
        Y = np.maximum(Y, 0.0)  # non negativity

    except Exception as e:
        print(f"[SKIP] Simulation failed for {fname}: {e}")
        continue
    nvars = Y.shape[0]
    names = get_names(mod, nvars)
    feats = compute_features(t, Y, OBS_WINDOW)
    model_label_guess = os.path.splitext(fname)[0]
    for i in range(nvars):
        row = {
            "model": model_label_guess,
            "species_index": i,
            "species_name": names[i],
        }
        row.update(feats[i])
        row["label"] = label_rule(row)
        rows.append(row)
    loaded_models.append(model_label_guess)
    print(f"[OK] {fname}: {nvars} species processed.")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")
print(f"Models included: {loaded_models}")
print(df.head(20))
