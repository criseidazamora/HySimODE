# =============================================================================
#  eval_lofo.py
# -----------------------------------------------------------------------------
#  Description:
#      Leave-One-Family-Out (LOFO) evaluation for the RFC trained on per-species
#      features. A "family" is defined by the prefix of the model name
#      (metab, signal, rl, grn, motif, osc, stochosc).
#
#      For each family F:
#         - Train on all species whose model is NOT in F
#         - Test on all species whose model IS in F
#         - Use the same hyperparameters as in the main LOMO evaluation
#           (best_params_group_aware from a LOMO summary JSON).
#
#  Outputs:
#      - lofo_metrics_by_family.csv : one row per family with metrics
#      - lofo_summary.json          : summary (means, std, coverage)
#
#  © 2026 Criseida G. Zamora Chimal
# =============================================================================

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight


# --- Misma whitelist de features que en train_rfc_lomo_v3_nested.py ---
FEATURES_ALLOWED = [
    "initial_q50", "initial_cv", "initial_minmax_ratio", "initial_nmadydt",
    "final_q50", "final_cv", "final_minmax_ratio", "final_nmadydt",
    "cv_total", "minmax_ratio_total", "nmadydt_total",
]


def _compute_class_weight_dict(y_train: np.ndarray):
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, cw)}


def _eval_binary(y_true, proba, threshold=0.5):
    """Same metric logic as in train_rfc_lomo_v3_nested.py (simplified)."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)

    y_pred = (proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    is_binary = (len(np.unique(y_true)) == 2)
    if is_binary:
        roc = roc_auc_score(y_true, proba)
        ap = average_precision_score(y_true, proba)
        mcc = matthews_corrcoef(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
    else:
        roc = ap = mcc = bacc = np.nan

    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc) if not np.isnan(bacc) else np.nan,
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc) if not np.isnan(roc) else np.nan,
        "average_precision": float(ap) if not np.isnan(ap) else np.nan,
        "mcc": float(mcc) if not np.isnan(mcc) else np.nan,
        "precision_1": float(prec[1]),
        "recall_1": float(rec[1]),
        "f1_1": float(f1[1]),
    }


def _infer_family(model_name: str) -> str:
    """
    Infer family from model string.
    Example:
        'metab01_linear_pathway' -> 'metab'
        'signal02_mapk...'       -> 'signal'
        'grn03_repressilator...' -> 'grn'
    """
    m = re.match(r"([a-zA-Z]+)", model_name)
    if m:
        return m.group(1)
    # Fallback: take prefix before first underscore
    return model_name.split("_")[0]


def main():
    ap = argparse.ArgumentParser(
        description="Leave-One-Family-Out (LOFO) evaluation for the RFC."
    )
    ap.add_argument("--csv", type=str, default="features_rfc.csv",
                    help="Features CSV (must contain 'model', 'label' and the RFC features).")
    ap.add_argument("--lomo_meta", type=str, default="lomo_summary.json",
                    help="LOMO summary JSON with 'best_params_group_aware'.")
    ap.add_argument("--out_csv", type=str, default="lofo_metrics_by_family.csv",
                    help="Output CSV with per-family metrics.")
    ap.add_argument("--out_json", type=str, default="lofo_summary.json",
                    help="Output JSON summary for LOFO.")
    ap.add_argument("--random_state", type=int, default=42,
                    help="Random seed for RF classifiers.")

    args = ap.parse_args()

    # ---------------------------------------------------------------------
    # 1) Cargar datos y definir familias
    # ---------------------------------------------------------------------
    df = pd.read_csv(args.csv)

    if "model" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'model' and 'label' columns.")

    # Enforce feature whitelist
    missing = [c for c in FEATURES_ALLOWED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    X_all = df.loc[:, FEATURES_ALLOWED].copy()
    y_all = df["label"].astype(int)
    models = df["model"].astype(str)

    # Compute family label
    families = models.apply(_infer_family)
    df["family"] = families

    unique_families = sorted(df["family"].unique())
    print(f"[LOFO] Families detected: {unique_families}")

    # ---------------------------------------------------------------------
    # 2) Cargar hiperparámetros desde el resumen LOMO
    # ---------------------------------------------------------------------
    with open(args.lomo_meta, "r") as f:
        meta = json.load(f)

    best_params = meta.get("best_params_group_aware")
    if best_params is None:
        raise KeyError(
            f"'best_params_group_aware' not found in {args.lomo_meta}. "
            "Run train_rfc_lomo_v3_nested.py in LOMO mode first to obtain it."
        )

    print(f"[LOFO] Using best_params_group_aware from {args.lomo_meta}:")
    print(best_params)

    # ---------------------------------------------------------------------
    # 3) Loop LOFO por familia
    # ---------------------------------------------------------------------
    rows = []
    for fam in unique_families:
        test_mask = (df["family"] == fam).values
        train_mask = ~test_mask

        X_train = X_all.iloc[train_mask]
        y_train = y_all.iloc[train_mask]
        X_test = X_all.iloc[test_mask]
        y_test = y_all.iloc[test_mask]

        if len(y_test) == 0:
            print(f"[LOFO] WARNING: family={fam} has no samples, skipping.")
            continue

        clf = RandomForestClassifier(
            n_jobs=-1,
            random_state=args.random_state,
            class_weight=_compute_class_weight_dict(y_train.values),
            **best_params,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]

        metrics = _eval_binary(y_test.values, proba, threshold=0.5)

        row = {
            "family": fam,
            "n_species": metrics["n"],
            "n_pos": metrics["n_pos"],
            "n_neg": metrics["n_neg"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "f1_macro": metrics["f1_macro"],
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["average_precision"],
            "mcc": metrics["mcc"],
            "precision_1": metrics["precision_1"],
            "recall_1": metrics["recall_1"],
            "f1_1": metrics["f1_1"],
        }
        rows.append(row)

        print(
            f"[LOFO] Family={fam:10s}  n={metrics['n']:3d}  "
            f"F1_1={metrics['f1_1']:.3f}  MCC={metrics['mcc'] if not np.isnan(metrics['mcc']) else float('nan'):.3f}  "
            f"BACC={metrics['balanced_accuracy'] if not np.isnan(metrics['balanced_accuracy']) else float('nan'):.3f}"
        )

    res = pd.DataFrame(rows)
    res.to_csv(args.out_csv, index=False)
    print(f"[LOFO] Saved per-family metrics: {args.out_csv}")

    # ---------------------------------------------------------------------
    # 4) Summary (mean/std + cobertura)
    # ---------------------------------------------------------------------
    metric_cols = [
        "accuracy", "balanced_accuracy", "f1_macro",
        "roc_auc", "average_precision", "mcc",
        "precision_1", "recall_1", "f1_1",
    ]

    summary = {
        "mode": "lofo",
        "csv": args.csv,
        "lomo_meta_used": args.lomo_meta,
        "n_families": int(len(res)),
        "families": list(res["family"].unique()),
        "metrics_mean": res[metric_cols].mean(numeric_only=True).to_dict(),
        "metrics_std": res[metric_cols].std(numeric_only=True).to_dict(),
        "metric_coverage": {
            "n_families_total": int(len(res)),
            "n_families_binary": int((res["n_pos"] > 0).mul(res["n_neg"] > 0).sum()),
            "n_families_monoclass": int((~((res["n_pos"] > 0) & (res["n_neg"] > 0))).sum()),
            **{f"n_defined_{c}": int(res[c].notna().sum()) for c in metric_cols},
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[LOFO] Saved summary: {args.out_json}")
    print("\n[LOFO] Mean metrics (family-level generalization):")
    for k, v in summary["metrics_mean"].items():
        std = summary["metrics_std"].get(k, float("nan"))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        print(f"  {k:>18s}: {float(v):.3f} ± {float(std):.3f}")


if __name__ == "__main__":
    main()
