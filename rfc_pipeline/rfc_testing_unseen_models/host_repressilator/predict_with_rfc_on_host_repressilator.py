# =============================================================================
#  predict_with_rfc_on_host_repressilator.py
# -----------------------------------------------------------------------------
#  Description:
#      Applies the trained and calibrated Random Forest Classifier (RFC) from
#      HySimODE to unseen trajectories of the host–repressilator model.
#      It loads the feature dataset, RFC model, and metadata (including the
#      optimal threshold for the stochastic class) to generate predictions for
#      each species.
#
#  Input files:
#      - features_host_repressilator.csv  : Features extracted from simulation
#      - rfc_calibrated.joblib            : Calibrated RFC model
#      - rfc_metadata.json                : Metadata with feature list and threshold
#
#  Output:
#      - predictions_host_repressilator.csv : CSV containing predicted labels and
#        probabilities for each species.
#
#  © 2025 Criseida G. Zamora Chimal. 
# =============================================================================

import json
import joblib
import pandas as pd

IN_CSV = "features_host_repressilator.csv"
MODEL_PATH = "rfc_calibrated.joblib"
META_PATH = "rfc_metadata.json"
OUT_CSV = "predictions_host_repressilator.csv"

def main():
    # 1) Load metadata and model
    meta = json.load(open(META_PATH, "r"))
    # From training: key "features"
    feature_names = meta.get("features") or meta.get("feature_names")
    if feature_names is None:
        raise KeyError("No 'features' or 'feature_names' key in metadata JSON.")
    clf = joblib.load(MODEL_PATH)

    # 2) Load new features
    df = pd.read_csv(IN_CSV)
    X = df[feature_names].copy()

    # 3) Prediction
    proba = clf.predict_proba(X)[:, 1]
    yhat = (proba >= meta.get("best_threshold_for_class1", 0.5)).astype(int)

    # 4) Output
    df["pred_label"] = yhat
    df["prob_stochastic"] = proba
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved predictions → {OUT_CSV}")
    print(df[["species_name", "pred_label", "prob_stochastic"]].head(15))

if __name__ == "__main__":
    main()
