# =============================================================================
#  predict_with_rfc_on_smolen.py
# -----------------------------------------------------------------------------
#  Description:
#      Applies the trained and calibrated Random Forest Classifier (RFC) from
#      HySimODE to unseen trajectories of the smolen model.
#      It loads the feature dataset, RFC model, and metadata (including the
#      optimal threshold for the stochastic class) to generate predictions for
#      each species.
#
#  Input files:
#      - features_smolen.csv  : Features extracted from simulation
#      - rfc_calibrated.joblib            : Calibrated RFC model
#      - rfc_metadata.json                : Metadata with feature list and threshold
#
#  Output:
#      - predictions_smolen.csv : CSV containing predicted labels and
#        probabilities for each species.
#
#  © 2025 Criseida G. Zamora Chimal. 
# =============================================================================
import json
import joblib
import pandas as pd

IN_CSV = "features_smolen.csv"              # or features_host_repressilator.csv
MODEL_PATH = "rfc_calibrated.joblib"
META_PATH = "rfc_metadata.json"
OUT_CSV = "predictions_smolen.csv"          # or predictions_host_repressilator.csv

def main():
    # 1) Load metadata and model
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    feature_names = meta.get("features") or meta.get("feature_names")
    if feature_names is None:
        raise KeyError("Metadata must include 'features' or 'feature_names' (list of column names).")

    clf = joblib.load(MODEL_PATH)

    # 2) Load new features
    df = pd.read_csv(IN_CSV)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in {IN_CSV}: {missing}")

    X = df[feature_names].copy()

    # 3) Prediction
    proba = clf.predict_proba(X)[:, 1]
    th = float(meta.get("best_threshold_for_class1", 0.5))
    yhat = (proba >= th).astype(int)

    # 4) Save outputs
    df["pred_label"] = yhat
    df["prob_stochastic"] = proba
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved predictions → {OUT_CSV}")
    if "species_name" in df.columns:
        print(df[["species_name", "pred_label", "prob_stochastic"]].head(15))
    else:
        print(df[["pred_label", "prob_stochastic"]].head(15))

if __name__ == "__main__":
    main()