# =============================================================================
#  rfc_integration.py
# -----------------------------------------------------------------------------
#  Description:
#      Core integration module that applies the calibrated Random Forest
#      Classifier (RFC) within HySimODE to partition model species into
#      stochastic and deterministic regimes.
#
#      This script computes the same 13 quantitative trajectory features
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
        y = np.maximum(Y[i, :], 0.0)  # same process that training
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
        final_mean = float(np.mean(yf))
        final_std = float(np.std(yf))
        final_cv = float(final_std / (final_mean + 1e-12))
        final_madydt = float(np.mean(np.abs(dyf)))
        final_value = float(y[-1])

        # Initial window
        yi = y[mask_initial]
        initial_mean = float(np.mean(yi))

        rows.append({
            "mean_total": mean_total,
            "std_total": std_total,
            "cv_total": cv_total,
            "madydt_total": madydt_total,
            "min": ymin, "max": ymax, "minmax_ratio": minmax_ratio,
            "final_mean": final_mean, "final_std": final_std,
            "final_cv": final_cv, "final_madydt": final_madydt,
            "final_value": final_value,
            "initial_mean": initial_mean,
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
    # Alinear columnas exactamente como en el entrenamiento
    X = feats.reindex(columns=feature_names, fill_value=0.0)

    proba = clf.predict_proba(X)[:, 1]
    labels = (proba >= prob_threshold).astype(int)

    decisions = feats.copy()
    decisions["prob_stochastic"] = proba
    decisions["pred_label"] = labels

    stochastic_indices = np.where(labels == 1)[0].tolist()
    return stochastic_indices, decisions

