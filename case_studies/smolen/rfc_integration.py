# =============================================================================
#  rfc_integration.py
# -----------------------------------------------------------------------------
#  Description:
#      Core integration module that applies the calibrated Random Forest
#      Classifier (RFC) within HySimODE to partition model species into
#      stochastic and deterministic regimes.
#
#      This script computes the same 11 quantitative trajectory features
#      used during RFC training and testing, ensuring full consistency
#      with the artifacts stored in `rfc_metadata.json`.
#
#  Pipeline consistency:
#      ✓ Feature definitions identical to: make_features_rfc.py
#      ✓ Training alignment with: train_rfc.py
#      ✓ Testing alignment with: predict_with_rfc_on_host_repressilator.py
#        and predict_with_rfc_on_smolen.py
#      ✓ Same observation and initial windows (100.0 / 50.0)
#      ✓ Same non-negativity preprocessing (np.maximum)
#
#  Notes:
#      - The RFC classifier and threshold are loaded directly from
#        `rfc_calibrated.joblib` and `rfc_metadata.json`.
#      - No manual rules, OOD checks, or heuristics are applied.
#      - This version is frozen to preserve reproducibility of the
#        published simulations and figures.
#
#  © 2025 Criseida G. Zamora Chimal
# =============================================================================

import json
import joblib
import numpy as np
import pandas as pd

# These windows must match those used during training
OBS_WINDOW = 100.0
INITIAL_WINDOW = 50.0
EPS = 1e-12

def _compute_features_from_timeseries(t: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """
    Computes the same features that were used to train the RFC
    t: (nt,)
    Y: (nvars, nt)
    """
    nvars = Y.shape[0]
    t_end = float(t[-1])
    mask_final = t >= (t_end - OBS_WINDOW)
    mask_initial = t <= INITIAL_WINDOW

    rows = []
    for i in range(nvars):
        y = np.maximum(Y[i, :], 0.0)  # same preprocessing used during training
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
    return pd.DataFrame(rows)

def load_rfc(model_path: str = "rfc_calibrated.joblib",
             meta_path: str = "rfc_metadata.json"):
    """
    Loads the calibrated model and metadata containing the feature columns
    and the recommended threshold.
    Returns: (clf, feature_names, best_threshold)
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)
    feature_names = meta.get("features") or meta.get("feature_names")
    if feature_names is None:
        raise KeyError("Metadata must include 'features' (list of column names).")
    best_th = meta.get("best_threshold_for_class1", 0.5)

    clf = joblib.load(model_path)
    return clf, feature_names, float(best_th)

def classify_species_with_rfc(t: np.ndarray,
                              Y: np.ndarray,
                              clf,
                              feature_names,
                              prob_threshold: float):
    """
    Applies the RFC without additional rules.
    Returns:
      - stochastic_indices: list of indices with prediction 1
      - decisions: DataFrame with features + prob_stochastic + pred_label
    """
    feats = _compute_features_from_timeseries(t, Y)
    # Align columns exactly as in training
    X = feats.reindex(columns=feature_names, fill_value=0.0)

    proba = clf.predict_proba(X)[:, 1]
    labels = (proba >= prob_threshold).astype(int)

    decisions = feats.copy()
    decisions["prob_stochastic"] = proba
    decisions["pred_label"] = labels

    stochastic_indices = np.where(labels == 1)[0].tolist()
    return stochastic_indices, decisions
