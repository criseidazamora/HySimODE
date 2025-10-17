# predict_with_rfc_on_smolen.py
import json
import joblib
import pandas as pd

IN_CSV = "features_smolen.csv"
MODEL_PATH = "rfc_calibrated.joblib"
META_PATH = "rfc_metadata.json"
OUT_CSV = "predictions_smolen.csv"

def main():
    meta = json.load(open(META_PATH, "r"))
    feature_names = meta.get("features") or meta.get("feature_names")
    if feature_names is None:
        raise KeyError("Metadata must include 'features' (list of column names).")
    clf = joblib.load(MODEL_PATH)

    df = pd.read_csv(IN_CSV)
    X = df.reindex(columns=feature_names, fill_value=0.0)

    proba = clf.predict_proba(X)[:, 1]
    th = float(meta.get("best_threshold_for_class1", 0.5))
    yhat = (proba >= th).astype(int)

    df["pred_label"] = yhat
    df["prob_stochastic"] = proba
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved predictions → {OUT_CSV}")
    print(df[["species_name", "pred_label", "prob_stochastic"]].head(15))

if __name__ == "__main__":
    main()
