# =============================================================================
# audit_family_similarity.py
# -----------------------------------------------------------------------------
# Utility for assessing similarity and potential redundancy among ODE models
# within the feature space used to build the HySimODE RFC training dataset.
#
# The script simulates each model, extracts the trajectory-based features used
# in the RFC pipeline, and compares models in that feature space. The purpose
# is to identify models with very similar dynamical profiles and assess whether
# closely related model families may be overrepresented in the dataset used to
# train and evaluate the classifier.
#
# © 2026 Criseida G. Zamora Chimal
# =============================================================================
import os
import re
import glob
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Helpers
# -----------------------------
def load_model_module(path):
    """Dynamically import a model .py file as a module."""
    modname = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robust_mad(x):
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12


def compute_species_features(t, y):
    """Compute per-species features from trajectory y(t)."""
    y = np.asarray(y, dtype=float)

    # Basic
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    eps = 1e-12

    # Label stats (fixed rule)
    q80 = float(np.quantile(y, 0.80))
    q99 = float(np.quantile(y, 0.99))
    label = int((q80 < 200.0) and (q99 < 200.0))

    # Dynamic features
    dy = np.gradient(y, t)
    nmad_dy = float(robust_mad(dy) / (robust_mad(y) + eps))

    # CV (avoid blow-up at tiny mean)
    cv = float(y_std / (abs(y_mean) + 1e-6))

    # Shape: minmax ratio (with safe denom)
    minmax_ratio = float((y_min + 1.0) / (y_max + 1.0))

    # Early vs late behavior (compare first 20% vs last 20%)
    n = len(y)
    k = max(5, int(0.2 * n))
    early = y[:k]
    late = y[-k:]
    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    early_late_ratio = float((early_mean + 1.0) / (late_mean + 1.0))

    # Log scale features (handle zeros)
    log_mean = float(np.log10(y_mean + 1.0))
    log_max = float(np.log10(y_max + 1.0))

    return {
        "q80": q80,
        "q99": q99,
        "label": label,
        "mean": y_mean,
        "std": y_std,
        "min": y_min,
        "max": y_max,
        "log_mean": log_mean,
        "log_max": log_max,
        "cv": cv,
        "minmax_ratio": minmax_ratio,
        "nMADdYdt": nmad_dy,
        "early_late_ratio": early_late_ratio,
    }


def model_signature(df_species, feature_cols):
    """
    Aggregate per-species features into a fixed-size model signature.
    Uses mean + (10th, 50th, 90th) percentiles across species.
    """
    sig = {}
    X = df_species[feature_cols].to_numpy()

    for j, col in enumerate(feature_cols):
        v = X[:, j]
        sig[f"{col}__mean"] = float(np.mean(v))
        sig[f"{col}__p10"] = float(np.quantile(v, 0.10))
        sig[f"{col}__p50"] = float(np.quantile(v, 0.50))
        sig[f"{col}__p90"] = float(np.quantile(v, 0.90))

    # Include label mix to reflect low/high-copy composition
    labels = df_species["label"].to_numpy()
    sig["label_frac_1"] = float(np.mean(labels))
    sig["n_species"] = int(len(labels))
    return sig


# -----------------------------
# Main audit
# -----------------------------
def simulate_model(module, t_eval=None):
    """Simulate model using required interface."""
    if t_eval is None:
        t_eval = np.linspace(module.TSPAN[0], module.TSPAN[1], 2000)

    # Prefer model_odes if present 
    if hasattr(module, "model_odes"):
        f = module.model_odes
    else:
        # fallback to dYdt(t, Y, p=PARAMS) signature
        def f(t, y):
            return module.dYdt(t, y, p=getattr(module, "PARAMS", None))

    sol = solve_ivp(
        f,
        module.TSPAN,
        np.array(module.Y0, dtype=float),
        t_eval=t_eval,
        method="Radau",
        rtol=1e-7,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {getattr(module, 'MODEL_NAME', 'unknown')}")

    return sol.t, sol.y


def audit_models(model_paths, outdir="audit_family", show_plots=False):
    os.makedirs(outdir, exist_ok=True)

    species_rows = []
    model_rows = []

    # Features used for similarity
    feature_cols = [
        "log_mean", "log_max", "cv", "minmax_ratio", "nMADdYdt", "early_late_ratio"
    ]

    for path in model_paths:
        module = load_model_module(path)
        model_name = getattr(module, "MODEL_NAME", os.path.basename(path))

        t, Y = simulate_model(module)
        names = getattr(module, "SPECIES_NAMES", [f"S{i}" for i in range(Y.shape[0])])

        # species-level features
        df_s = []
        for i, sp in enumerate(names):
            feats = compute_species_features(t, Y[i])
            feats.update({
                "model": model_name,
                "file": os.path.basename(path),
                "species": sp,
                "idx": i,
            })
            species_rows.append(feats)
            df_s.append(feats)

        df_s = pd.DataFrame(df_s)
        sig = model_signature(df_s, feature_cols)
        sig.update({"model": model_name, "file": os.path.basename(path)})
        model_rows.append(sig)

    df_species = pd.DataFrame(species_rows)
    df_models = pd.DataFrame(model_rows)

    # Save species-level
    species_csv = os.path.join(outdir, "audit_species_family.csv")
    df_species.to_csv(species_csv, index=False)

    # Build similarity on model signatures
    sig_cols = [c for c in df_models.columns if c not in ("model", "file")]
    X = df_models[sig_cols].to_numpy(dtype=float)

    Xs = StandardScaler().fit_transform(X)
    S = cosine_similarity(Xs)

    # Save similarity matrix
    sim_df = pd.DataFrame(S, index=df_models["model"], columns=df_models["model"])
    sim_csv = os.path.join(outdir, "model_similarity_cosine.csv")
    sim_df.to_csv(sim_csv)

    # Report top redundant pairs (excluding diagonal)
    pairs = []
    for i in range(S.shape[0]):
        for j in range(i + 1, S.shape[1]):
            pairs.append((df_models["model"][i], df_models["model"][j], float(S[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    topk = min(15, len(pairs))
    top_pairs_df = pd.DataFrame(pairs[:topk], columns=["model_a", "model_b", "cosine_sim"])
    top_pairs_csv = os.path.join(outdir, "top_redundant_pairs.csv")
    top_pairs_df.to_csv(top_pairs_csv, index=False)

    # Nearest neighbor per model
    nn_rows = []
    for i, name in enumerate(df_models["model"]):
        sims = S[i].copy()
        sims[i] = -np.inf
        j = int(np.argmax(sims))
        nn_rows.append({
            "model": name,
            "nearest_model": df_models["model"][j],
            "cosine_sim": float(S[i, j]),
        })
    nn_df = pd.DataFrame(nn_rows).sort_values("cosine_sim", ascending=False)
    nn_csv = os.path.join(outdir, "nearest_neighbor_per_model.csv")
    nn_df.to_csv(nn_csv, index=False)

    # Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(S, aspect="auto")
    plt.colorbar(label="cosine similarity (model signatures)")
    plt.xticks(range(len(df_models)), df_models["model"], rotation=90, fontsize=7)
    plt.yticks(range(len(df_models)), df_models["model"], fontsize=7)
    plt.tight_layout()
    heatmap_png = os.path.join(outdir, "model_similarity_heatmap.png")
    plt.savefig(heatmap_png, dpi=220)
    if show_plots:
        plt.show()
    plt.close()

    # PCA of model signatures
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    plt.figure(figsize=(7, 5))
    plt.scatter(Z[:, 0], Z[:, 1])
    for i, name in enumerate(df_models["model"]):
        plt.text(Z[i, 0], Z[i, 1], name, fontsize=7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of model signatures (family subset)")
    plt.tight_layout()
    pca_png = os.path.join(outdir, "pca_models.png")
    plt.savefig(pca_png, dpi=220)
    if show_plots:
        plt.show()
    plt.close()

    # Console summary
    print(f"\nWrote:\n- {species_csv}\n- {sim_csv}\n- {top_pairs_csv}\n- {nn_csv}\n- {heatmap_png}\n- {pca_png}")
    print("\nTop redundant pairs:")
    print(top_pairs_df.to_string(index=False))
    print("\nMost redundant models by nearest-neighbor similarity:")
    print(nn_df.head(10).to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", type=str, default=None, help='Glob pattern e.g. "metab*.py"')
    ap.add_argument("--models", type=str, nargs="*", default=None, help="Explicit list of model files")
    ap.add_argument("--outdir", type=str, default="audit_family", help="Output directory")
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    args = ap.parse_args()

    if args.models:
        paths = args.models
    elif args.glob:
        paths = sorted(glob.glob(args.glob))
    else:
        raise SystemExit("Provide --glob or --models")

    if len(paths) < 2:
        raise SystemExit("Need at least 2 model files for similarity.")

    audit_models(paths, outdir=args.outdir, show_plots=args.show)
