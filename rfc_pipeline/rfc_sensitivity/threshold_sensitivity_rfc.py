# =============================================================================
# threshold_sensitivity_rfc.py
# -----------------------------------------------------------------------------
# Re-evaluates the sensitivity of the HySimODE Random Forest classifier (RFC)
# to the labeling threshold T while keeping the entire training pipeline fixed.
#
# Fixed components:
#     - same feature dataset (features_rfc.csv)
#     - same quantile-based labeling rule (q80, q99), with variable threshold T
#     - same statistical unit: ODE models evaluated via LOMO
#     - same globally selected hyperparameters (best_params_group_aware),
#       loaded from a JSON file
#
# Outputs:
#     threshold_sensitivity_by_model.csv
#         columns:
#         [T, held_out_model, n_species, n_pos, n_neg,
#          balanced_accuracy, mcc]
#
#     threshold_sensitivity_summary.json
#         for each threshold T:
#             - n_species_total
#             - n_label_flips_vs_T200 (and percentage)
#             - balanced_accuracy_mean / std
#             - mcc_mean / std
#
#         plus metadata (CSV path, JSON file from which hyperparameters
#         were loaded, etc.).

# Example usage:
#
# python threshold_sensitivity_rfc.py \
#     --features_csv features_rfc.csv \
#     --trace_csv features_rfc_trace.csv \
#     --lomo_summary_json lomo_summary_bootstrap.json \
#     --thresholds 150,200,250,300
#
# Required inputs:
#     features_rfc.csv
#     features_rfc_trace.csv
#     lomo_summary_bootstrap.json
#
# Outputs:
#     threshold_sensitivity_by_model.csv
#     threshold_sensitivity_summary.json
#
# © 2025-2026 Criseida G. Zamora Chimal
# =============================================================================

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------------------------------------------
# Default configuration (consistent with train_rfc.py and make_features_rfc.py)
# --------------------------------------------------------------------------------------

FEATURES_ALLOWED: List[str] = [
    "initial_q50",
    "initial_cv",
    "initial_minmax_ratio",
    "initial_nmadydt",
    "final_q50",
    "final_cv",
    "final_minmax_ratio",
    "final_nmadydt",
    "cv_total",
    "minmax_ratio_total",
    "nmadydt_total",
]

MERGE_KEYS = ["model", "species_index", "species_name"]


# --------------------------------------------------------------------------------------
# Utilities: metrics and class-weight computation (reused from train_rfc.py)
# --------------------------------------------------------------------------------------

def _compute_class_weight_dict(y_train: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, cw)}


def _eval_binary(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Replicates the _eval_binary logic used in train_rfc.py."""
    y_pred = (proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    # Per-class metrics (always computed with zero_division=0)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    # Macro F1 score (always defined)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

   # Metrics that require both classes to be present
    is_binary = (len(np.unique(y_true)) == 2)
    roc = roc_auc_score(y_true, proba) if is_binary else np.nan
    ap = average_precision_score(y_true, proba) if is_binary else np.nan
    mcc = matthews_corrcoef(y_true, y_pred) if is_binary else np.nan
    bacc = balanced_accuracy_score(y_true, y_pred) if is_binary else np.nan

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc) if not np.isnan(bacc) else np.nan,
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc) if not np.isnan(roc) else np.nan,
        "average_precision": float(ap) if not np.isnan(ap) else np.nan,
        "mcc": float(mcc) if not np.isnan(mcc) else np.nan,
        "precision_0": float(prec[0]),
        "recall_0": float(rec[0]),
        "f1_0": float(f1[0]),
        "precision_1": float(prec[1]),
        "recall_1": float(rec[1]),
        "f1_1": float(f1[1]),
        "threshold": float(threshold),
        "is_binary": bool(is_binary),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# --------------------------------------------------------------------------------------
# Data loading and label recomputation
# --------------------------------------------------------------------------------------

def load_merged_dataframe(features_csv: str, trace_csv: str) -> pd.DataFrame:
    """Loads features_rfc.csv and features_rfc_trace.csv and merges them."""
    df_feat = pd.read_csv(features_csv)
    df_trace = pd.read_csv(trace_csv)

    expected_trace_cols = MERGE_KEYS + ["label_q80", "label_q99"]
    missing_cols = [c for c in expected_trace_cols if c not in df_trace.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns {missing_cols} in trace CSV '{trace_csv}'. "
            f"Available: {list(df_trace.columns)}"
        )

    df = pd.merge(
        df_feat,
        df_trace[expected_trace_cols],
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )

    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in features CSV '{features_csv}'.")

    if "model" not in df.columns:
        raise ValueError(f"'model' column not found in features CSV '{features_csv}'.")

    # Ensure proper data types
    df["label"] = df["label"].astype(int)
    df["model"] = df["model"].astype(str)

    return df


def relabel_for_threshold(df: pd.DataFrame, T: float) -> pd.Series:
    """
    Applies the same labeling rule used in make_features_rfc.py, but with
    a variable threshold T:

        label_T = 1 if (q80 < T) AND (q99 < T); 0 otherwise.

    The quantiles are read from the columns 'label_q80' and 'label_q99'.
    """
    if "label_q80" not in df.columns or "label_q99" not in df.columns:
        raise ValueError("DataFrame must contain 'label_q80' and 'label_q99' columns.")

    y_T = ((df["label_q80"] < T) & (df["label_q99"] < T)).astype(int)
    return y_T


def compute_label_flips(df: pd.DataFrame, y_T: pd.Series, base_label_col: str = "label") -> Dict[str, float]:
    """Computes the number and percentage of species that change label relative to T=200."""
    base = df[base_label_col].astype(int).values
    new = y_T.astype(int).values
    if base.shape[0] != new.shape[0]:
        raise ValueError("Base labels and new labels must have the same length.")
    flips = (base != new)
    n_flips = int(flips.sum())
    pct_flips = 100.0 * float(n_flips) / float(len(base))
    return {"n_flips": n_flips, "pct_flips": pct_flips}


# --------------------------------------------------------------------------------------
# # LOMO evaluation with fixed hyperparameters
# --------------------------------------------------------------------------------------

def run_lomo_for_T(
    df: pd.DataFrame,
    y_T: pd.Series,
    best_params: Dict,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Runs LOMO evaluation by model using:
        - X = FEATURES_ALLOWED
        - y = y_T
        - groups = df["model"]
        - hyperparameters = best_params (best_params_group_aware)
        - class_weight='balanced', recomputed for each fold
        - decision threshold = 0.5

    Returns a DataFrame containing per-model evaluation metrics.
    """
    # Extract X, y, and groups
    missing_feats = [c for c in FEATURES_ALLOWED if c not in df.columns]
    if missing_feats:
        raise ValueError(
            f"Missing feature columns {missing_feats} in merged DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.loc[:, FEATURES_ALLOWED].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN detected in feature matrix after numeric coercion. Bad columns: {bad}")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite values detected in feature matrix.")

    y = y_T.astype(int).values
    groups = df["model"].astype(str).values

    unique_models = np.unique(groups)
    records = []

    for m in sorted(unique_models):
        test_mask = (groups == m)
        train_mask = ~test_mask

        X_train, y_train = X.iloc[train_mask], y[train_mask]
        X_test, y_test = X.iloc[test_mask], y[test_mask]

        # Balanced class weights computed per fold
        class_weight = _compute_class_weight_dict(y_train)

        clf = RandomForestClassifier(
            n_jobs=-1,
            random_state=random_state,
            class_weight=class_weight,
            **best_params,
        )
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        metrics = _eval_binary(y_test, proba, threshold=0.5)

        records.append(
            {
                "T": None,  
                "held_out_model": str(m),
                "n_species": int(len(y_test)),
                "n_pos": int(y_test.sum()),
                "n_neg": int((1 - y_test).sum()),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "mcc": float(metrics["mcc"]),
            }
        )

    res = pd.DataFrame(records)
    
    res["T"] = float("nan")  
    return res


# --------------------------------------------------------------------------------------
# # Load hyperparameters from JSON
# --------------------------------------------------------------------------------------

def load_best_params_from_json(path: str) -> Dict:
    """
    Loads best_params_group_aware from a LOMO summary JSON file
    (e.g., lomo_summary.json, lomo_summary_bootstrap.json, etc.).
    """
    with open(path, "r") as f:
        data = json.load(f)

    if "best_params_group_aware" not in data:
        raise KeyError(
            f"'best_params_group_aware' not found in JSON '{path}'. "
            "Make sure you are passing a LOMO summary JSON."
        )

    best_params = data["best_params_group_aware"]
    #  Normalize parameter keys 
    allowed_keys = {
        "n_estimators",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "max_depth",
        "bootstrap",
    }
    filtered = {k: v for k, v in best_params.items() if k in allowed_keys}

    return filtered


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Threshold-sensitivity analysis for HySimODE RFC (LOMO + fixed best_params)."
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        default="features_rfc.csv",
        help="CSV with features and base labels (output of make_features_rfc.py).",
    )
    parser.add_argument(
        "--trace_csv",
        type=str,
        default="features_rfc_trace.csv",
        help="CSV with label_q80 and label_q99 (output of make_features_rfc.py).",
    )
    parser.add_argument(
        "--lomo_summary_json",
        type=str,
        default="lomo_summary_bootstrap.json",
        help="JSON file containing 'best_params_group_aware' (e.g., lomo_summary.json or lomo_summary_bootstrap.json).",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="150,200,250,300",
        help="Comma-separated list of T values (molecule-count thresholds) to test.",
    )
    parser.add_argument(
        "--out_by_model_csv",
        type=str,
        default="threshold_sensitivity_by_model.csv",
        help="Output CSV with per-model metrics for each T.",
    )
    parser.add_argument(
        "--out_summary_json",
        type=str,
        default="threshold_sensitivity_summary.json",
        help="Output JSON with aggregated statistics per T.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for RandomForestClassifier.",
    )

    args = parser.parse_args()

    # Parse thresholds
    T_values: List[float] = []
    for tok in args.thresholds.split(","):
        tok = tok.strip()
        if not tok:
            continue
        T_values.append(float(tok))

    print(f"[INFO] Threshold-sensitivity analysis for T = {T_values}")

    # Load merged dataset
    df = load_merged_dataframe(args.features_csv, args.trace_csv)
    n_species_total = int(len(df))
    print(f"[INFO] Loaded {n_species_total} species from '{args.features_csv}'.")

    # Load globally selected hyperparameters
    best_params = load_best_params_from_json(args.lomo_summary_json)
    print(f"[INFO] Loaded best_params_group_aware from '{args.lomo_summary_json}': {best_params}")

    # Initialize result containers
    all_rows_by_model = []
    summary_per_T = {}

    # Baseline labels (original T=200) used to compare label flips
    base_labels = df["label"].astype(int).values

    for T in T_values:
        print(f"\n[INFO] Processing threshold T = {T:.1f} ...")

        # 1) Re-labelling
        y_T = relabel_for_threshold(df, T)

        # 2) Flips vs T=200
        flip_info = compute_label_flips(df, y_T, base_label_col="label")
        print(
            f"    Label flips vs T=200: {flip_info['n_flips']} / {n_species_total} "
            f"({flip_info['pct_flips']:.2f}%)"
        )

        # 3) LOMO with fixed hyperparameters
        res_T = run_lomo_for_T(df, y_T, best_params, random_state=args.random_state)
        res_T["T"] = float(T)  # fix T value
        all_rows_by_model.append(res_T)

        # 4) Summary statistics
        bacc_mean = float(res_T["balanced_accuracy"].mean(skipna=True))
        bacc_std = float(res_T["balanced_accuracy"].std(skipna=True))
        mcc_mean = float(res_T["mcc"].mean(skipna=True))
        mcc_std = float(res_T["mcc"].std(skipna=True))

        summary_per_T[str(T)] = {
            "T": float(T),
            "n_species_total": n_species_total,
            "n_label_flips_vs_T200": int(flip_info["n_flips"]),
            "pct_label_flips_vs_T200": float(flip_info["pct_flips"]),
            "balanced_accuracy_mean": bacc_mean,
            "balanced_accuracy_std": bacc_std,
            "mcc_mean": mcc_mean,
            "mcc_std": mcc_std,
        }

        print(
            f"    BACC (mean ± std): {bacc_mean:.3f} ± {bacc_std:.3f} | "
            f"MCC (mean ± std): {mcc_mean:.3f} ± {mcc_std:.3f}"
        )

    # --- Save results ---

    by_model_df = pd.concat(all_rows_by_model, ignore_index=True)
    by_model_df.to_csv(args.out_by_model_csv, index=False)
    print(f"\n[INFO] Saved per-model metrics to '{args.out_by_model_csv}'.")

    summary = {
        "features_csv": args.features_csv,
        "trace_csv": args.trace_csv,
        "lomo_summary_json_used": args.lomo_summary_json,
        "best_params_group_aware": best_params,
        "thresholds_tested": T_values,
        "per_threshold": summary_per_T,
    }

    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved summary statistics to '{args.out_summary_json}'.")
    print("\nDone.")


if __name__ == "__main__":
    main()
