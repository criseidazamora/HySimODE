# =============================================================================
# train_rfc.py
# -----------------------------------------------------------------------------
# Description:
#     Training pipeline for the Random Forest classifier used in HySimODE to
#     partition model species into stochastic and deterministic simulation
#     regimes.
#
#     The classifier is trained on per-species trajectory features extracted
#     from deterministic simulations of biochemical ODE models. Each row of
#     the dataset corresponds to one species described by statistical features
#     computed from its time series.
#
#     Model performance is evaluated using Leave-One-Model-Out (LOMO)
#     cross-validation, where all species from one ODE model are held out for
#     testing while the classifier is trained on species from the remaining
#     models. This protocol measures generalization to previously unseen
#     biochemical systems.
#
# Outputs:
#     - rfc_calibrated.joblib : calibrated Random Forest classifier
#     - rfc_metadata.json     : feature names and decision threshold
#
# Example usage:
#     python train_rfc.py --csv features_rfc.csv --mode lomo --random_state 42 \
#         --out_lomo_csv lomo_metrics_by_model.csv \
#         --out_lomo_json lomo_summary.json
#
# Notes:
#     Feature definitions must match those used in:
#         - make_features_rfc.py
#         - rfc_integration.py
#
# © 2025-2026 Criseida G. Zamora Chimal
# =============================================================================

import argparse
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


DROP_COLS_DEFAULT = ["model", "species_name", "species_index", "label"]
FEATURES_ALLOWED = [
    "initial_q50", "initial_cv", "initial_minmax_ratio", "initial_nmadydt",
    "final_q50", "final_cv", "final_minmax_ratio", "final_nmadydt",
    "cv_total", "minmax_ratio_total", "nmadydt_total",
]


def _load_xy(csv_path: str, drop_cols=None, features_allowed=None):
    df = pd.read_csv(csv_path)

    if drop_cols is None:
        drop_cols = DROP_COLS_DEFAULT
    if features_allowed is None:
        features_allowed = FEATURES_ALLOWED

    # Required columns
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column (0/1).")
    if "model" not in df.columns:
        raise ValueError("CSV must contain a 'model' column for grouped evaluation (LOMO).")

    y = df["label"].astype(int)
    groups = df["model"].astype(str)

    # Enforce whitelist (prevents legacy/leakage columns from ever entering training)
    missing = [c for c in features_allowed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.loc[:, features_allowed].copy()

    # Ensure numeric and finite
    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN detected in features after numeric coercion. Columns with NaN: {bad}")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite values (inf/-inf) detected in feature matrix.")

    if X.shape[1] == 0:
        raise ValueError("No feature columns found (empty X).")

    return df, X, y, groups



def _compute_class_weight_dict(y_train: np.ndarray):
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, cw)}


def _fit_random_search(X_train, y_train, groups_train=None, n_iter=40, random_state=42, n_splits=5):
    """Group-aware hyperparameter search if groups_train is provided."""
    base = RandomForestClassifier(
        n_jobs=-1,
        random_state=random_state,
        class_weight=_compute_class_weight_dict(y_train),
    )

    param_dist = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [None, 6, 10, 14, 20],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": ["sqrt", "log2", 0.5, 0.7, None],
        "bootstrap": [True, False],
    }

    if groups_train is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="matthews_corrcoef",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        search.fit(X_train, y_train)
    else:
        # Grouped CV for model-level generalization
        cv = GroupKFold(n_splits=n_splits)
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="matthews_corrcoef",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        search.fit(X_train, y_train, groups=groups_train)

    return search.best_estimator_, search.best_params_


def _eval_binary(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    # Per-class metrics (always returned, even if a class is absent in y_true)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )

    # Macro metrics are defined even for single-class y_true (sklearn uses zero_division rules)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Threshold-free metrics and MCC require both classes in y_true to be meaningful
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
        "y_pred": y_pred,
    }

def _best_threshold_for_f1(y_true, proba, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    best_th = 0.5
    best_f1 = -1.0
    for th in grid:
        y_pred = (proba >= th).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
        f1_pos = float(f1[1])
        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_th = float(th)
    return best_th, best_f1


def run_standard(args):
    df, X, y, groups = _load_xy(args.csv)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    best_rfc, best_params = _fit_random_search(
        X_train, y_train, groups_train=None, n_iter=args.n_iter, random_state=args.random_state, n_splits=5
    )
    print("\nBest params:", best_params)

    # Validation (uncalibrated)
    proba_valid = best_rfc.predict_proba(X_valid)[:, 1]
    m0 = _eval_binary(y_valid.values, proba_valid, threshold=0.5)

    print("\n=== Validation (threshold=0.5) ===")
    print(
        f"Accuracy: {m0['accuracy']:.3f}  ROC-AUC: {m0['roc_auc']:.3f}  AP: {m0['average_precision']:.3f}"
    )
    print(
        f"Class 1 (stochastic): Precision={m0['precision_1']:.3f} Recall={m0['recall_1']:.3f} F1={m0['f1_1']:.3f}"
    )
    print("\nConfusion matrix:\n", np.array(m0["confusion_matrix"]))
    print("\nClassification report:\n", classification_report(y_valid, m0["y_pred"], digits=3))

    # Threshold selection on validation
    best_th, best_f1 = _best_threshold_for_f1(y_valid.values, proba_valid)
    print(f"\nBest threshold for class 1 F1: {best_th:.2f} (F1={best_f1:.3f})")

    # Calibration (standard mode only)
    calibrator = CalibratedClassifierCV(best_rfc, cv=5, method="isotonic")
    calibrator.fit(X_train, y_train)
    proba_valid_cal = calibrator.predict_proba(X_valid)[:, 1]
    m_cal = _eval_binary(y_valid.values, proba_valid_cal, threshold=best_th)

    print("\n=== Validation (calibrated, threshold=best_th) ===")
    print(
        f"Accuracy: {m_cal['accuracy']:.3f}  ROC-AUC: {m_cal['roc_auc']:.3f}  AP: {m_cal['average_precision']:.3f}"
    )
    print(
        f"Class 1 (stochastic): Precision={m_cal['precision_1']:.3f} Recall={m_cal['recall_1']:.3f} F1={m_cal['f1_1']:.3f}"
    )
    print("\nConfusion matrix:\n", np.array(m_cal["confusion_matrix"]))

    # Feature importance (from uncalibrated best_rfc)
    importances = pd.Series(best_rfc.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 15 feature importances:\n", importances.head(15))

    # Save calibrated model + metadata
    if not args.no_save:
        joblib.dump(calibrator, args.out_model)
        with open(args.out_meta, "w") as f:
            json.dump(
                {
                    "mode": "standard",
                    "csv": args.csv,
                    "best_params": best_params,
                    "best_threshold_for_class1": best_th,
                    "features": list(X.columns),
                    "metrics_valid_calibrated": {
                        "accuracy": m_cal["accuracy"],
                        "roc_auc": m_cal["roc_auc"],
                        "average_precision": m_cal["average_precision"],
                        "f1_class1": m_cal["f1_1"],
                    },
                },
                f,
                indent=2,
            )

        print(f"\nSaved model: {args.out_model}")
        print(f"Saved metadata: {args.out_meta}")


def run_lomo(args):
    df, X, y, groups = _load_xy(args.csv)

    # ---------------------------------------------------------------------
    # (A) Global group-aware hyperparameter search (used for FINAL model)
    # ---------------------------------------------------------------------
    print("\n[LOMO] Group-aware hyperparameter search (GroupKFold) ...")
    best_rfc_global, best_params = _fit_random_search(
        X,
        y,
        groups_train=groups,
        n_iter=args.n_iter,
        random_state=args.random_state,
        n_splits=min(5, groups.nunique()),
    )
    print("\n[LOMO] Best params (group-aware):", best_params)

    # ---------------------------------------------------------------------
    # (B) Standard LOMO evaluation with fixed global hyperparameters
    # ---------------------------------------------------------------------
    unique_models = sorted(groups.unique())
    rows = []

    for m in unique_models:
        test_mask = (groups == m).values
        train_mask = ~test_mask

        X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
        X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]

        clf = RandomForestClassifier(
            n_jobs=-1,
            random_state=args.random_state,
            class_weight=_compute_class_weight_dict(y_train.values),
            **best_params,
        )
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        metrics = _eval_binary(y_test.values, proba, threshold=0.5)

        rows.append(
            {
                "held_out_model": m,
                "n_species": int(len(y_test)),
                "n_pos": int(y_test.sum()),
                "n_neg": int((1 - y_test).sum()),
                **{k: metrics[k] for k in [
                    "accuracy","balanced_accuracy","f1_macro",
                    "roc_auc","average_precision","mcc",
                    "precision_1","recall_1","f1_1"
                ]},
            }
        )

        if args.verbose_lomo:
            def _fmt(x):
                return "nan" if (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
            print(
                f"[LOMO] Hold-out={m:>30s}  n={len(y_test):3d}  "
                f"F1_1={_fmt(metrics['f1_1'])}  MCC={_fmt(metrics['mcc'])}  "
                f"BACC={_fmt(metrics['balanced_accuracy'])}  "
                f"ROC={_fmt(metrics['roc_auc'])}  AP={_fmt(metrics['average_precision'])}"
            )

    res = pd.DataFrame(rows)

    metric_cols = [
        "accuracy","balanced_accuracy","f1_macro",
        "roc_auc","average_precision","mcc",
        "precision_1","recall_1","f1_1",
    ]

    summary = {
        "mode": "lomo",
        "csv": args.csv,
        "n_models": int(len(unique_models)),
        "best_params_group_aware": best_params,
        "metrics_mean": res[metric_cols].mean(numeric_only=True).to_dict(),
        "metrics_std": res[metric_cols].std(numeric_only=True).to_dict(),
        "metric_coverage": {
            "n_models_total": int(len(res)),
            "n_models_binary": int((res["n_pos"] > 0).mul(res["n_neg"] > 0).sum()),
            "n_models_monoclass": int((~((res["n_pos"] > 0) & (res["n_neg"] > 0))).sum()),
            **{f"n_defined_{c}": int(res[c].notna().sum()) for c in metric_cols},
        },
        "hardest_models_by_f1_1": res.sort_values("f1_1", ascending=True).head(min(10, len(res)))[
            ["held_out_model", "n_species", "f1_1", "mcc"]
        ].to_dict(orient="records"),
        "hardest_models_by_mcc": res.sort_values("mcc", ascending=True).head(min(10, len(res)))[
            ["held_out_model", "n_species", "mcc", "f1_1"]
        ].to_dict(orient="records"),
    }

    # ---------------------------------------------------------------------
    # (C) Optional: Nested LOMO (outer = held-out model, inner = GroupKFold on training models)
    # ---------------------------------------------------------------------
    nested_res = None
    nested_best_params_by_fold = None
    if getattr(args, "nested_lomo", False):
        print("\n[LOMO] Nested LOMO enabled (more conservative): inner GroupKFold search repeated per held-out model ...")
        nested_rows = []
        nested_best_params_by_fold = {}

        for m in unique_models:
            test_mask = (groups == m).values
            train_mask = ~test_mask

            X_train, y_train = X.iloc[train_mask], y.iloc[train_mask]
            X_test, y_test = X.iloc[test_mask], y.iloc[test_mask]
            groups_train = groups.iloc[train_mask]

            _, best_fold_params = _fit_random_search(
                X_train,
                y_train,
                groups_train=groups_train,
                n_iter=getattr(args, "nested_n_iter", args.n_iter),
                random_state=args.random_state,
                n_splits=min(5, groups_train.nunique()),
            )
            nested_best_params_by_fold[m] = best_fold_params

            clf = RandomForestClassifier(
                n_jobs=-1,
                random_state=args.random_state,
                class_weight=_compute_class_weight_dict(y_train.values),
                **best_fold_params,
            )
            clf.fit(X_train, y_train)

            proba = clf.predict_proba(X_test)[:, 1]
            metrics = _eval_binary(y_test.values, proba, threshold=0.5)

            nested_rows.append(
                {
                    "held_out_model": m,
                    "n_species": int(len(y_test)),
                    "n_pos": int(y_test.sum()),
                    "n_neg": int((1 - y_test).sum()),
                    **{k: metrics[k] for k in [
                        "accuracy","balanced_accuracy","f1_macro",
                        "roc_auc","average_precision","mcc",
                        "precision_1","recall_1","f1_1"
                    ]},
                }
            )

            if args.verbose_lomo:
                def _fmt(x):
                    return "nan" if (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
                print(
                    f"[NESTED-LOMO] Hold-out={m:>30s}  n={len(y_test):3d}  "
                    f"F1_1={_fmt(metrics['f1_1'])}  MCC={_fmt(metrics['mcc'])}  "
                    f"BACC={_fmt(metrics['balanced_accuracy'])}"
                )

        nested_res = pd.DataFrame(nested_rows)
        nested_csv = getattr(args, "out_nested_csv", "nested_lomo_metrics_by_model.csv")
        nested_res.to_csv(nested_csv, index=False)

        summary["nested_lomo"] = {
            "enabled": True,
            "nested_n_iter": int(getattr(args, "nested_n_iter", args.n_iter)),
            "per_model_csv": nested_csv,
            "best_params_by_fold": nested_best_params_by_fold,
            "metrics_mean": nested_res[metric_cols].mean(numeric_only=True).to_dict(),
            "metrics_std": nested_res[metric_cols].std(numeric_only=True).to_dict(),
            "metric_coverage": {
                "n_models_total": int(len(nested_res)),
                "n_models_binary": int((nested_res["n_pos"] > 0).mul(nested_res["n_neg"] > 0).sum()),
                "n_models_monoclass": int((~((nested_res["n_pos"] > 0) & (nested_res["n_neg"] > 0))).sum()),
                **{f"n_defined_{c}": int(nested_res[c].notna().sum()) for c in metric_cols},
            },
        }

    # ---------------------------------------------------------------------
    # (D) Bootstrap CIs (treat models as independent units)
    # ---------------------------------------------------------------------
    if args.bootstrap_ci:
        summary["bootstrap_ci_over_models"] = _bootstrap_ci_over_models(
            res,
            metric_cols=metric_cols,
            n_boot=args.n_boot,
            random_state=args.random_state,
        )

    # ---------------------------------------------------------------------
    # (E) Permutation test (null = permuted labels; default = within-model)
    # ---------------------------------------------------------------------
    if args.permutation_test:
        print(f"[LOMO] Entering permutation test block (n_perm={args.n_perm})", flush=True)

        perm = _permutation_test_lomo(
            X,
            y,
            groups,
            best_params,
            n_perm=args.n_perm,
            within_model=not args.permute_global,
            random_state=args.random_state,
        )
        if not args.save_perm_samples:
            perm.pop("null_samples", None)
        summary["permutation_test"] = perm

    # ---------------------------------------------------------------------
    # Save LOMO outputs
    # ---------------------------------------------------------------------
    res.to_csv(args.out_lomo_csv, index=False)
    with open(args.out_lomo_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[LOMO] Saved per-model metrics: {args.out_lomo_csv}")
    print(f"[LOMO] Saved summary: {args.out_lomo_json}")

    print("\n[LOMO] Mean metrics (unseen-model generalization):")
    for k, v in summary["metrics_mean"].items():
        std = summary["metrics_std"].get(k, float("nan"))
        vv = float(v) if v is not None else float("nan")
        ss = float(std) if std is not None else float("nan")
        if isinstance(vv, float) and np.isnan(vv):
            continue
        print(f"  {k:>18s}: {vv:.3f} ± {ss:.3f}")

    if "nested_lomo" in summary:
        print("\n[NESTED-LOMO] Mean metrics (more conservative estimate):")
        for k, v in summary["nested_lomo"]["metrics_mean"].items():
            std = summary["nested_lomo"]["metrics_std"].get(k, float("nan"))
            vv = float(v) if v is not None else float("nan")
            ss = float(std) if std is not None else float("nan")
            if isinstance(vv, float) and np.isnan(vv):
                continue
            print(f"  {k:>18s}: {vv:.3f} ± {ss:.3f}")

    # ---------------------------------------------------------------------
    # Optional: train final calibrated model on ALL data for deployment
    # ---------------------------------------------------------------------
    if args.train_final and (not args.no_save):
        print("\n[LOMO] Training FINAL calibrated model on all data (for deployment) ...")

        final_base = RandomForestClassifier(
            n_jobs=-1,
            random_state=args.random_state,
            class_weight=_compute_class_weight_dict(y.values),
            **best_params,
        )
        final_base.fit(X, y)

        calibrator = CalibratedClassifierCV(final_base, cv=5, method="isotonic")
        calibrator.fit(X, y)

        joblib.dump(calibrator, args.out_model)

        with open(args.out_meta, "w") as f:
            json.dump(
                {
                    "mode": "trained_after_lomo",
                    "csv": args.csv,
                    "best_params_group_aware": best_params,
                    "features": list(X.columns),
                    "best_threshold_for_class1": 0.5,
                    "lomo_summary": summary,
                },
                f,
                indent=2,
            )
        print(f"[LOMO] Saved final calibrated model: {args.out_model}")
        print(f"[LOMO] Saved metadata: {args.out_meta}")

def _bootstrap_ci_over_models(res: pd.DataFrame, metric_cols, n_boot=2000, random_state=0):
    """Bootstrap CIs treating held-out models as the independent units."""
    rng = np.random.default_rng(random_state)
    vals = {m: [] for m in metric_cols}
    models = res["held_out_model"].values
    n = len(res)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)  # sample models with replacement
        samp = res.iloc[idx]
        for m in metric_cols:
            vals[m].append(float(np.nanmean(samp[m].values)))
    ci = {}
    for m in metric_cols:
        arr = np.array(vals[m], dtype=float)
        ci[m] = {
            "mean_boot": float(np.nanmean(arr)),
            "ci95_low": float(np.nanpercentile(arr, 2.5)),
            "ci95_high": float(np.nanpercentile(arr, 97.5)),
        }
    return ci


def _permute_labels(y: pd.Series, groups: pd.Series, within_model: bool, random_state: int):
    rng = np.random.default_rng(random_state)
    y_perm = y.copy()
    if within_model:
        for g in groups.unique():
            mask = (groups == g).values
            y_perm.loc[mask] = rng.permutation(y_perm.loc[mask].values)
    else:
        y_perm[:] = rng.permutation(y_perm.values)
    return y_perm


def _permutation_test_lomo(X: pd.DataFrame, y: pd.Series, groups: pd.Series, best_params: dict, *,
                           n_perm: int = 200, within_model: bool = True, random_state: int = 0):
    """Permutation test under the LOMO protocol, keeping hyperparameters fixed.

    Returns a dict with null distributions and p-values for key metrics.
    """
    rng = np.random.default_rng(random_state)
    unique_models = sorted(groups.unique())
    print(f"[Permutation] START n_perm={n_perm} within_model={within_model}", flush=True)


    def _run_once(y_use: pd.Series, seed: int):
        per_model = []
        for m in unique_models:
            test_mask = (groups == m).values
            train_mask = ~test_mask
            X_train, y_train = X.iloc[train_mask], y_use.iloc[train_mask]
            X_test, y_test = X.iloc[test_mask], y_use.iloc[test_mask]

            clf = RandomForestClassifier(
                n_jobs=-1,
                random_state=seed,
                class_weight=_compute_class_weight_dict(y_train.values),
                **best_params,
            )
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)[:, 1]
            metrics = _eval_binary(y_test.values, proba, threshold=0.5)
            per_model.append(metrics)
        # macro-average across held-out models (each model counts equally)
        f1_1 = np.nanmean([m["f1_1"] for m in per_model])
        mcc = np.nanmean([m["mcc"] for m in per_model])
        acc = np.nanmean([m["accuracy"] for m in per_model])
        return float(acc), float(f1_1), float(mcc)

    # observed
    obs_acc, obs_f1, obs_mcc = _run_once(y, seed=int(rng.integers(1, 1_000_000)))

    null_acc, null_f1, null_mcc = [], [], []
    for i in range(int(n_perm)):
        if (i + 1) % 5 == 0:   
            print(f"[Permutation] Completed {i+1}/{n_perm}", flush=True)
        y_perm = _permute_labels(y, groups, within_model=within_model, random_state=int(rng.integers(1, 1_000_000)))
        a, f, m = _run_once(y_perm, seed=int(rng.integers(1, 1_000_000)))
        null_acc.append(a)
        null_f1.append(f)
        null_mcc.append(m)

    def pval(null, obs):
        null = np.array(null, dtype=float)
        
        return float((np.sum(null >= obs) + 1.0) / (len(null) + 1.0))

    return {
        "n_perm": int(n_perm),
        "within_model": bool(within_model),
        "observed": {"accuracy": obs_acc, "f1_1": obs_f1, "mcc": obs_mcc},
        "null_mean": {"accuracy": float(np.mean(null_acc)), "f1_1": float(np.mean(null_f1)), "mcc": float(np.mean(null_mcc))},
        "null_std": {"accuracy": float(np.std(null_acc)), "f1_1": float(np.std(null_f1)), "mcc": float(np.std(null_mcc))},
        "p_value": {"accuracy": pval(null_acc, obs_acc), "f1_1": pval(null_f1, obs_f1), "mcc": pval(null_mcc, obs_mcc)},
        "null_samples": {
            "accuracy": null_acc,
            "f1_1": null_f1,
            "mcc": null_mcc,
        },
    }


def main():
    p = argparse.ArgumentParser(description="Train RFC for stochastic/deterministic per-species classification.")
    p.add_argument("--csv", type=str, default="features_rfc.csv", help="Path to features CSV (must include model,label).")
    p.add_argument("--mode", choices=["standard", "lomo"], default="standard", help="Evaluation mode.")
    p.add_argument("--test_size", type=float, default=0.25, help="Validation split fraction (standard mode).")
    p.add_argument("--n_iter", type=int, default=40, help="RandomizedSearchCV iterations.")
    p.add_argument("--random_state", type=int, default=42, help="Random seed.")
    p.add_argument("--no_save", action="store_true", help="Do not write joblib/json outputs.")
    p.add_argument("--out_model", type=str, default="rfc_calibrated.joblib", help="Output joblib filename.")
    p.add_argument("--out_meta", type=str, default="rfc_metadata.json", help="Output metadata JSON filename.")

    # LOMO outputs
    p.add_argument("--out_lomo_csv", type=str, default="lomo_metrics_by_model.csv", help="Per-model LOMO metrics CSV.")
    p.add_argument("--out_lomo_json", type=str, default="lomo_summary.json", help="LOMO summary JSON.")
    p.add_argument("--verbose_lomo", action="store_true", help="Print per-held-out-model metrics.")
    p.add_argument("--train_final", action="store_true", help="After LOMO, train and save final calibrated model on ALL data.")

    # Nested LOMO (evaluation only; does not change final hyperparams)
    p.add_argument("--nested_lomo", action="store_true", help="Run nested LOMO (inner GroupKFold hyperparameter search per held-out model).")
    p.add_argument("--nested_n_iter", type=int, default=20, help="RandomizedSearchCV iterations for the INNER search in nested LOMO.")
    p.add_argument("--out_nested_csv", type=str, default="nested_lomo_metrics_by_model.csv", help="Per-model nested LOMO metrics CSV.")

    # Robustness / statistical validation (LOMO mode)
    p.add_argument("--bootstrap_ci", action="store_true", help="Compute bootstrap 95 percent CIs over held-out models (LOMO).")
    p.add_argument("--n_boot", type=int, default=2000, help="Number of bootstrap resamples (LOMO).")
    p.add_argument("--permutation_test", action="store_true", help="Run permutation test under LOMO (hyperparams fixed).")
    p.add_argument("--n_perm", type=int, default=200, help="Number of permutations for permutation test.")
    p.add_argument("--permute_global", action="store_true", help="Permute labels globally (default permutes within each model).")
    p.add_argument("--save_perm_samples", action="store_true", help="Store all null samples in JSON (can be large).")

    args = p.parse_args()

    if args.mode == "standard":
        run_standard(args)
    else:
        run_lomo(args)


if __name__ == "__main__":
    main()
