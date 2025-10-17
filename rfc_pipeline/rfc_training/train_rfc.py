# =============================================================================
#  train_rfc.py
# -----------------------------------------------------------------------------
#  Description:
#      Trains and calibrates a Random Forest Classifier (RFC) to partition
#      biochemical model variables into stochastic or deterministic regimes.
#      The training data are computed quantitative features per species,
#      extracted from deterministic simulations.
#
#      This script performs:
#        1. Data loading and preprocessing
#        2. Stratified train/validation split
#        3. Class weighting for imbalance
#        4. Randomized hyperparameter search
#        5. Model evaluation (default threshold = 0.5)
#        6. Optimal threshold search for the stochastic class (max F1)
#        7. Probability calibration (isotonic)
#        8. Feature importance inspection
#        9. Export of the calibrated model and training metadata
#
#  Output files:
#      - rfc_calibrated.joblib   : Calibrated RFC model
#      - rfc_metadata.json       : Metadata (features, best params, threshold, metrics)
#
#  © 2025 Criseida G. Zamora Chimal
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# === 1) Load dataset ===
CSV_PATH = "features_rfc.csv"   # <-- rename if needed
df = pd.read_csv(CSV_PATH)

# Drop non-numeric identifiers that must not be used as features:
drop_cols = ["model", "species_name", "species_index", "label"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
y = df["label"].astype(int)

# Ensure X contains only numeric columns
X = X.select_dtypes(include=[np.number]).copy()

# === 2) Stratified train/validation split ===
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# === 3) Class weights to handle imbalance ===
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

# === 4) Randomized hyperparameter search (fast) ===
rfc = RandomForestClassifier(
    n_jobs=-1,
    class_weight=class_weight_dict,
    random_state=42
)

param_dist = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [None, 6, 10, 14, 20],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2", 0.5, 0.7, None],
    "bootstrap": [True, False]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    rfc,
    param_distributions=param_dist,
    n_iter=40,
    scoring="f1",            # optimize overall F1 score (could use 'f1_macro' or 'average_precision')
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
search.fit(X_train, y_train)

best_rfc = search.best_estimator_
print("\nBest params:", search.best_params_)

# === 5) Validation metrics (threshold = 0.5) ===
proba_valid = best_rfc.predict_proba(X_valid)[:, 1]
y_pred = (proba_valid >= 0.5).astype(int)

acc = accuracy_score(y_valid, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_valid, y_pred, average=None, labels=[0,1])
f1_pos = f1[1]  # F1 for stochastic class
roc = roc_auc_score(y_valid, proba_valid)
ap = average_precision_score(y_valid, proba_valid)

print("\n=== Validation (threshold=0.5) ===")
print(f"Accuracy: {acc:.3f}  ROC-AUC: {roc:.3f}  AP: {ap:.3f}")
print(f"Class 1 (stochastic): Precision={prec[1]:.3f} Recall={rec[1]:.3f} F1={f1_pos:.3f}")
print("\nConfusion matrix:\n", confusion_matrix(y_valid, y_pred))
print("\nClassification report:\n", classification_report(y_valid, y_pred, digits=3))

# === 6) Optimal threshold suggestion for class 1 (max F1 for y=1) ===
ths = np.linspace(0.05, 0.95, 19)
f1_scores = []
for th in ths:
    yp = (proba_valid >= th).astype(int)
    _, _, f1_tmp, _ = precision_recall_fscore_support(y_valid, yp, average=None, labels=[0,1])
    f1_scores.append(f1_tmp[1])
best_th = float(ths[int(np.argmax(f1_scores))])
best_f1 = float(np.max(f1_scores))
print(f"\nBest threshold for class 1 F1: {best_th:.2f} (F1={best_f1:.3f})")

# === 7) (Optional) Probability calibration ===
# Useful if non-0.5 thresholds are applied or if well-calibrated probabilities are needed.
calibrator = CalibratedClassifierCV(best_rfc, cv=5, method="isotonic")
calibrator.fit(X_train, y_train)
proba_valid_cal = calibrator.predict_proba(X_valid)[:, 1]
y_pred_cal = (proba_valid_cal >= best_th).astype(int)

acc_c = accuracy_score(y_valid, y_pred_cal)
prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(y_valid, y_pred_cal, average=None, labels=[0,1])
roc_c = roc_auc_score(y_valid, proba_valid_cal)
ap_c = average_precision_score(y_valid, proba_valid_cal)
print("\n=== Validation (calibrated, threshold=best_th) ===")
print(f"Accuracy: {acc_c:.3f}  ROC-AUC: {roc_c:.3f}  AP: {ap_c:.3f}")
print(f"Class 1 (stochastic): Precision={prec_c[1]:.3f} Recall={rec_c[1]:.3f} F1={f1_c[1]:.3f}")
print("\nConfusion matrix:\n", confusion_matrix(y_valid, y_pred_cal))

# === 8) Feature importances ===
importances = pd.Series(best_rfc.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 feature importances:\n", importances.head(15))

# === 9) Save model + metadata (with recommended threshold) ===
joblib.dump(calibrator, "rfc_calibrated.joblib")   # save calibrated RFC
with open("rfc_metadata.json", "w") as f:
    json.dump({
        "best_params": search.best_params_,
        "best_threshold_for_class1": best_th,
        "features": list(X.columns),
        "metrics_valid": {
            "accuracy": float(acc_c),
            "roc_auc": float(roc_c),
            "average_precision": float(ap_c),
            "f1_class1": float(f1_c[1])
        }
    }, f, indent=2)

print("\nSaved model: rfc_calibrated.joblib")
print("Saved metadata: rfc_metadata.json")
