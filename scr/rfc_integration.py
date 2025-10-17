# rfc_integration.py — versión minimal, estrictamente basada en RFC
# No OOD, no reglas manuales, no zonas ambiguas.

import json
import joblib
import numpy as np
import pandas as pd

# Estas ventanas deben coincidir con las usadas en el entrenamiento
OBS_WINDOW = 100.0
INITIAL_WINDOW = 50.0

def _compute_features_from_timeseries(t: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
    """
    Calcula las mismas features que se usaron para entrenar el RFC.
    t: (nt,)
    Y: (nvars, nt)
    """
    nvars = Y.shape[0]
    t_end = float(t[-1])
    mask_final = t >= (t_end - OBS_WINDOW)
    mask_initial = t <= INITIAL_WINDOW

    rows = []
    for i in range(nvars):
        y = np.maximum(Y[i, :], 0.0)  # mismo preprocesado que en training
        dy = np.gradient(y, t)

        # Trayectoria completa
        mean_total = float(np.mean(y))
        std_total = float(np.std(y))
        cv_total = float(std_total / (mean_total + 1e-12))
        madydt_total = float(np.mean(np.abs(dy)))
        ymin, ymax = float(np.min(y)), float(np.max(y))
        minmax_ratio = float((ymin + 1e-12) / (ymax + 1e-12))

        # Ventana final
        yf = y[mask_final]
        dyf = dy[mask_final]
        final_mean = float(np.mean(yf))
        final_std = float(np.std(yf))
        final_cv = float(final_std / (final_mean + 1e-12))
        final_madydt = float(np.mean(np.abs(dyf)))
        final_value = float(y[-1])

        # Ventana inicial
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
    Carga el modelo calibrado y la metadata con columnas y umbral recomendado.
    Devuelve: (clf, feature_names, best_threshold)
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
    Aplica el RFC sin reglas adicionales.
    Devuelve:
      - stochastic_indices: lista de índices con predicción 1
      - decisions: DataFrame con features + prob_stochastic + pred_label
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
