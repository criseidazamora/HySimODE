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
#      (Legacy, retained for traceability)
#      mean_total, std_total, cv_total, madydt_total,
#      min, max, minmax_ratio,
#      final_mean, final_std, final_cv, final_madydt,
#      final_value, initial_mean
#
#      (Robust windowed features used in the revised benchmark)
#      initial_q50, initial_cv, initial_minmax_ratio, initial_nmadydt,
#      final_q50, final_cv, final_minmax_ratio, final_nmadydt,
#      cv_total_rep, minmax_ratio_rep, nmadydt_rep
#
#  Labeling rule:
#      A species is labeled as stochastic (1) if it stays in the low-copy regime
#      both typically and in rare-event tails:
#          (q0.80(y) < 200) AND (q0.99(y) < 200)
#      Quantiles are computed over the full simulated trajectory.
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

    # ======================
    # Family 1 — Metabolism
    # ======================
    "metab01_linear_pathway.py",
    "metab02_linear_pathway_MM.py",
    "metab03_linear_pathway_feedback.py",
    "metab04_reduced_glycolysis.py",
    "metab05_glycolysis_intermediate.py",
    "metab06_glycolysis_pulsed_input.py",


    # ======================================
    # Family 2 — Phosphorylation / Signaling
    # ======================================
    "signal01_mapk_single_layer.py",
    "signal02_mapk_double_phosphorylation.py",
    "signal03_mapk_with_phosphatase.py",
    "signal04_goldbeter_koshland.py",
    "signal05_mapk_three_layer.py",
    "signal06_mapk_three_layer_feedback.py",

    # ======================================
    # Family 3 — Receptor/Ligand Systems
    # ======================================
    "rl01_ligand_receptor_binding.py",
    "rl02_receptor_activation.py",
    "rl03_receptor_internalization.py",
    "rl04_ligand_degradation.py",

    # ===============================
    # Family 4 — Deterministic Oscillators
    # ===============================
    "osc01_lotka_volterra.py",
    "osc02_lotka_volterra_logistic.py",
    "osc03_goodwin_oscillator.py",
    "osc04_goodwin_delay_chain.py",

    # ===============================
    # Family 5 — Gene Regulatory Circuits
    # ===============================
    "grn01_repressilator.py",
    "grn02_repressilator_saturable_deg.py",
    "grn03_repressilator_dimerization.py",
    "grn04_toggle_switch.py",
    "grn05_toggle_autoregulation_v2_balanced.py",
    "grn06_toggle_asymmetric.py",
    "grn07_lambda_switch.py",
    "grn08_lambda_switch_cooperative.py",


    # ===============================
    # Family 6 — Motif-Hybrid
    # ===============================
    "motif01_i1ffl_incoherent.py",
    "motif02_i1ffl_coherent.py",
    "motif03_i1ffl_saturable_output.py",
    "motif04_stochastic_gene_deterministic_protein.py",
    "motif05_stochastic_enzyme_deterministic.py",
    "motif06_delayed_negative_feedback_pulse.py",


    # ===============================
    # Family 7 — Stochastic Oscillators
    # ===============================
    "stochosc01_brusselator.py",
    "stochosc02_oregonator_reduced.py",
    "stochosc03_activator_inhibitor.py",
    "stochosc04_fitzhugh_nagumo_like.py",
    "stochosc05_brusselator_saturable_deg.py",
    "stochosc06_activator_inhibitor_pulsed_input.py"
]

T_FINAL = 2000.0
N_POINTS = 1000
OBS_WINDOW = 100.0
INITIAL_WINDOW = 50.0

# -----------------------------------------------------------------------------
# Label definition (scale-based, conservative for hybrid/SSA feasibility)
# -----------------------------------------------------------------------------
# We label a species as "stochastic" (label=1) only if it stays in the low-copy
# regime not just typically (q80), but also without rare high-abundance bursts (q99).
# This guards against short-lived but very large transients that would make SSA
# steps prohibitively expensive due to extremely large propensities.
LABEL_Q_TYPICAL = 0.80   # Typical-regime quantile
LABEL_Q_SAFETY  = 0.99   # Safety quantile to exclude rare high-copy bursts
LABEL_THRESHOLD = 200.0  # Molecule-count threshold for regime assignment

# -----------------------------------------------------------------------------
# "Representative" interval for global features
# -----------------------------------------------------------------------------
# For consistency with the label (which is evaluated over the full trajectory),
# we compute representative ("_rep") global features over the full simulated
# horizon. Windowed features (initial/final) remain informative about early
# transients and late-time behavior.
EPS = 1e-12
ATOL = 1e-9
RTOL = 1e-7
METHOD = "Radau"

OUT_CSV = "features_rfc.csv"
OUT_TRACE_CSV = "features_rfc_trace.csv"

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
    """Compute quantitative features for each species.

    This function returns a dictionary per species containing:
      (i) legacy features retained for traceability, and
      (ii) robust windowed/normalized features used in the revised benchmark.

    The label is computed separately by `label_rule()` using a conservative quantile-based criterion (q80 and q99).
    """
    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)
    t_end = float(t[-1])

    # Feature windows
    mask_final = t >= (t_end - obs_window)
    mask_initial = t <= INITIAL_WINDOW

    feats: List[Dict[str, Any]] = []
    for i in range(Y.shape[0]):
        y = Y[i, :].astype(float)

        # Time derivative
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

        feats.append({
            # ---- Robust windowed features (recommended for training) -------
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
    return feats


def label_rule(t: np.ndarray, y: np.ndarray) -> int:
    """Binary labeling rule (scale-based, conservative): 1 = stochastic, 0 = deterministic.

    We label a species as stochastic (label=1) only if it remains in the low-copy regime
    both typically and in rare-event tails. Concretely, we require:

        (q_{0.80}(y) < 200) AND (q_{0.99}(y) < 200)

    where quantiles are computed over the full simulated trajectory. The additional
    high-quantile constraint prevents classifying species with short-lived but very
    large bursts as stochastic, which would make SSA/hybrid simulations prohibitively
    expensive due to extremely large propensities.
    """
    y = np.asarray(y, dtype=float)

    q_typ = float(np.quantile(y, LABEL_Q_TYPICAL))
    q_saf = float(np.quantile(y, LABEL_Q_SAFETY))
    return int((q_typ < LABEL_THRESHOLD) and (q_saf < LABEL_THRESHOLD))


# ---------- Main run ----------
rows = []
trace_rows = []
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
        row["label"] = label_rule(t, Y[i, :])

        rows.append(row)
        y_full = Y[i, :].astype(float)
        trace_rows.append({
            "model": model_label_guess,
            "species_index": i,
            "species_name": names[i],
            "label": row["label"],
            "label_q80": float(np.quantile(y_full, LABEL_Q_TYPICAL)),
            "label_q99": float(np.quantile(y_full, LABEL_Q_SAFETY)),
        })

    loaded_models.append(model_label_guess)
    print(f"[OK] {fname}: {nvars} species processed.")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

df_trace = pd.DataFrame(trace_rows)
df_trace.to_csv(OUT_TRACE_CSV, index=False)

print(f"\nSaved: {OUT_CSV}")
print(f"\nSaved: {OUT_TRACE_CSV}")
print(f"Models included: {loaded_models}")
print(df.head(20))