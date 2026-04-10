# =============================================================================
# audit_models.py
# -----------------------------------------------------------------------------
# Scientific audit and cross-domain validation utility for an ODE benchmark dataset.
#
# What it does:
# 1) Discovers model *.py files in a directory, dynamically imports them,
#    and runs deterministic ODE simulations.
#
# 2) Computes per-species diagnostics including q80/q99 labels:
#       label = 1 if (q80 < 200) and (q99 < 200); else 0
#
# 3) Extracts the compact feature set used in the RFC benchmark:
#       - cv_total_rep, minmax_ratio_rep, nmadydt_rep
#       - initial_* features over the first INITIAL_WINDOW minutes
#       - final_* features over the last FINAL_WINDOW minutes
#
# 4) Produces:
#       audit_inventory.csv
#       audit_species.csv
#       audit_report.md
#       cv_by_model.csv
#       cv_by_family.csv
#       optional PCA plot and redundancy diagnostics
#
# Usage:
#     python audit_models.py --models_dir . --t_final 3000 --n_points 2000
#
# Notes:
# - Each model module must define:
#       Y0 : initial conditions
#       *_odes(t, y) : ODE function
#       optional var_names : species names
#
# © 2026 Criseida G. Zamora Chimal
# =============================================================================

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Optional (for plots). Script still works without matplotlib.
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

# Optional (for ML validation)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
    )
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# ----------------------------- Label definition ------------------------------
LABEL_Q_TYPICAL = 0.80   # q80
LABEL_Q_SAFETY  = 0.99   # q99
LABEL_THRESHOLD = 200.0

# --------------------------- Feature window settings -------------------------
INITIAL_WINDOW = 50.0    # minutes
FINAL_WINDOW   = 100.0   # minutes

# -------------------------- ODE integration defaults -------------------------
DEFAULT_T_FINAL  = 2000.0
DEFAULT_N_POINTS = 1000
SOLVER_METHOD    = "Radau"
RTOL             = 1e-7
ATOL             = 1e-9

# ------------------------------ File discovery -------------------------------
DEFAULT_EXCLUDE = {
    "audit_models.py",
    "make_features_rfc.py",
    "make_features_rfc_updated.py",
    "make_features_rfc_updated_v2.py",
    "make_features_rfc_corrected.py",
    "train_rfc.py",
}

FAMILY_PREFIX = [
    ("metab", "Family 1 — Core Metabolism / Smooth Dynamics"),
    ("signal", "Family 2 — Phosphorylation & Signaling Cascades"),
    ("rl", "Family 3 — Receptor–Ligand Binding and Trafficking"),
    ("osc", "Family 4 — Deterministic Oscillators"),
    ("grn", "Family 5 — Gene Regulatory Circuits"),
    ("motif", "Family 6 — Network Motifs / Multiscale"),
    ("stochosc", "Family 7 — Noise-Prone Oscillatory Models"),

]

# ----------------------------- Helper functions ------------------------------

def infer_family(filename: str) -> str:
    base = Path(filename).name.lower()
    for pref, fam in FAMILY_PREFIX:
        if base.startswith(pref):
            return fam
    return "Uncategorized"


def dynamic_import(py_path: Path):
    """Import a model python file as a module."""
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  
    return mod


def get_ode_fn(mod) -> Optional[Callable]:
    """Find and return the ODE function inside an imported model module."""
    
    for attr in dir(mod):
        if attr.endswith("_odes") and callable(getattr(mod, attr)):
            return getattr(mod, attr)
    return None


def get_var_names(mod, nvars: int) -> List[str]:
    """Return variable names if defined in the module; otherwise use generic labels."""
    if hasattr(mod, "var_names"):
        try:
            names = list(getattr(mod, "var_names"))
            if len(names) == nvars:
                return names
        except Exception:
            pass
    return [f"Y{i}" for i in range(nvars)]


def simulate_module(mod, t_final: float, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the ODE model using solve_ivp."""
    if not hasattr(mod, "Y0"):
        raise RuntimeError("Model missing global Y0.")
    y0 = np.array(mod.Y0, dtype=float)
    odes = get_ode_fn(mod)
    if odes is None:
        raise RuntimeError("No ODE function found (expected *_odes).")

    t_span = (0.0, float(t_final))
    t_eval = np.linspace(t_span[0], t_span[1], int(n_points))

    sol = solve_ivp(
        odes,
        t_span,
        y0,
        t_eval=t_eval,
        method=SOLVER_METHOD,
        rtol=RTOL,
        atol=ATOL,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    Y = np.asarray(sol.y, dtype=float)

    # Numerical cleanup: no negative molecule counts
    Y[Y < 0.0] = 0.0
    return np.asarray(sol.t, dtype=float), Y


def safe_quantile(y: np.ndarray, q: float) -> float:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.quantile(y, q))


def compute_label(y: np.ndarray) -> Tuple[int, float, float]:
    """Compute label + diagnostics (q80, q99) over the full trajectory."""
    q80 = safe_quantile(y, LABEL_Q_TYPICAL)
    q99 = safe_quantile(y, LABEL_Q_SAFETY)
    label = 1 if (q80 < LABEL_THRESHOLD and q99 < LABEL_THRESHOLD) else 0
    return label, q80, q99


def cv(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = float(np.mean(y))
    sd = float(np.std(y))
    return float(sd / (mu + 1e-12))


def minmax_ratio(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    return float((ymin + 1e-12) / (ymax + 1e-12))


def nmadydt(y: np.ndarray, dy: np.ndarray) -> float:
    """Normalized mean absolute derivative: E[|dy/dt|] / (median(y)+eps)."""
    med = float(np.median(y))
    return float(np.mean(np.abs(dy)) / (med + 1e-12))


def finite_diff(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """dy/dt estimate aligned to y (same length) using numpy gradient."""
    return np.gradient(y, t)


def species_features(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute the compact feature set + label diagnostics for one species."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy = finite_diff(t, y)

    # Full-trajectory ("rep") features
    cv_rep = cv(y)
    mm_rep = minmax_ratio(y)
    nmad_rep = nmadydt(y, dy)

    # Initial window
    t0 = float(t[0])
    mask_i = t <= (t0 + INITIAL_WINDOW)
    yi = y[mask_i]
    dyi = dy[mask_i]
    initial_q50 = float(np.median(yi)) if yi.size else float("nan")
    initial_cv = cv(yi) if yi.size else float("nan")
    initial_mm = minmax_ratio(yi) if yi.size else float("nan")
    initial_nmad = nmadydt(yi, dyi) if yi.size else float("nan")

    # Final window
    t_end = float(t[-1])
    mask_f = t >= (t_end - FINAL_WINDOW)
    yf = y[mask_f]
    dyf = dy[mask_f]
    final_q50 = float(np.median(yf)) if yf.size else float("nan")
    final_cv = cv(yf) if yf.size else float("nan")
    final_mm = minmax_ratio(yf) if yf.size else float("nan")
    final_nmad = nmadydt(yf, dyf) if yf.size else float("nan")

    # Oscillation diagnostic (simple, scale-normalized amplitude)
    q50 = float(np.median(y))
    amp = float(np.max(y) - np.min(y))
    osc_metric = float(amp / (q50 + 1e-12))

    # Label + label diagnostics
    label, q80, q99 = compute_label(y)

    return {
        # --- Training features  ---
        "cv_total": cv_rep,
        "minmax_ratio_total": mm_rep,
        "nmadydt_total": nmad_rep,
        "initial_q50": initial_q50,
        "initial_cv": initial_cv,
        "initial_minmax_ratio": initial_mm,
        "initial_nmadydt": initial_nmad,
        "final_q50": final_q50,
        "final_cv": final_cv,
        "final_minmax_ratio": final_mm,
        "final_nmadydt": final_nmad,
        # --- Diagnostics ---
        "label": float(label),
        "label_q80": q80,
        "label_q99": q99,
        "q50": q50,
        "amp": amp,
        "osc_metric": osc_metric,
    }


def is_model_file(path: Path, exclude: set[str]) -> bool:
    if path.suffix != ".py":
        return False
    if path.name in exclude:
        return False
    if path.name.startswith("_"):
        return False
    return True


def build_dataset(models_dir: Path, t_final: float, n_points: int, exclude: set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate all models and build (species_df, inventory_df)."""
    rows = []
    inv_rows = []

    model_files = sorted([p for p in models_dir.glob("*.py") if is_model_file(p, exclude)])

    for p in model_files:
        model_name = p.name
        family = infer_family(model_name)
        try:
            mod = dynamic_import(p)
            t, Y = simulate_module(mod, t_final=t_final, n_points=n_points)
            nvars = int(Y.shape[0])
            names = get_var_names(mod, nvars)
            if len(names) != nvars:
                names = [f"Y{i}" for i in range(nvars)]

            labels = []
            q99s = []
            osc_counts = 0

            for i in range(nvars):
                feats = species_features(t, Y[i, :])
                label = int(feats["label"])
                labels.append(label)
                q99s.append(feats["label_q99"])
                if feats["osc_metric"] > 0.25:
                    osc_counts += 1

                row = {
                    "model": model_name,
                    "family": family,
                    "species_index": i,
                    "species": names[i],
                }
                row.update(feats)
                rows.append(row)

            labels = np.asarray(labels, dtype=int)
            label1 = int(np.sum(labels == 1))
            label0 = int(np.sum(labels == 0))
            frac1 = float(label1 / max(1, nvars))

            q99s = np.asarray(q99s, dtype=float)
            inv_rows.append({
                "model": model_name,
                "family": family,
                "n_species": nvars,
                "label1_count": label1,
                "label0_count": label0,
                "label1_fraction": frac1,
                "q99_min": float(np.nanmin(q99s)) if q99s.size else float("nan"),
                "q99_median": float(np.nanmedian(q99s)) if q99s.size else float("nan"),
                "q99_max": float(np.nanmax(q99s)) if q99s.size else float("nan"),
                "oscillatory_species_count": osc_counts,
            })

        except Exception as e:
            inv_rows.append({
                "model": model_name,
                "family": family,
                "n_species": float("nan"),
                "label1_count": float("nan"),
                "label0_count": float("nan"),
                "label1_fraction": float("nan"),
                "q99_min": float("nan"),
                "q99_median": float("nan"),
                "q99_max": float("nan"),
                "oscillatory_species_count": float("nan"),
                "error": str(e),
            })
            continue

    species_df = pd.DataFrame(rows)
    inv_df = pd.DataFrame(inv_rows)
    return species_df, inv_df


def compute_redundancy(X: np.ndarray) -> Dict[str, float]:
    """Compute a simple nearest-neighbor redundancy score in standardized feature space."""
    
    n = X.shape[0]
    if n < 3:
        return {"nn_min": float("nan"), "nn_median": float("nan"), "nn_p05": float("nan")}
    # Compute squared distances 
    G = X @ X.T
    sq = np.sum(X**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * G
    np.fill_diagonal(D2, np.inf)
    nn = np.sqrt(np.min(np.maximum(D2, 0.0), axis=1))
    return {
        "nn_min": float(np.min(nn)),
        "nn_median": float(np.median(nn)),
        "nn_p05": float(np.quantile(nn, 0.05)),
    }


def group_cv(species_df: pd.DataFrame, group_col: str, feature_cols: List[str], seed: int = 0) -> pd.DataFrame:
    """Cross-domain validation using group-aware CV (by model or by family)."""
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn not available in this environment.")

    df = species_df.dropna(subset=feature_cols + ["label", group_col]).copy()
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df[group_col].astype(str).to_numpy()

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise RuntimeError(f"Not enough groups for {group_col} CV (need >=2).")

    # Choose splitter
    if n_groups >= 5:
        splitter = GroupKFold(n_splits=5)
        splits = splitter.split(X, y, groups=groups)
    else:
        splitter = LeaveOneGroupOut()
        splits = splitter.split(X, y, groups=groups)

    # Model (match RFC intent)
    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
    )

    out_rows = []
    fold = 0
    for tr, te in splits:
        fold += 1
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        g_te = groups[te]

        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)

        out_rows.append({
            "fold": fold,
            "group_col": group_col,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_groups_test": int(len(np.unique(g_te))),
            "accuracy": float(accuracy_score(yte, yhat)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, yhat)),
            "precision": float(precision_score(yte, yhat, zero_division=0)),
            "recall": float(recall_score(yte, yhat, zero_division=0)),
            "f1": float(f1_score(yte, yhat, zero_division=0)),
            "mcc": float(matthews_corrcoef(yte, yhat)) if len(np.unique(yte)) > 1 else float("nan"),
        })
    return pd.DataFrame(out_rows)


def random_split_baseline(species_df: pd.DataFrame, feature_cols: List[str], seed: int = 0) -> Dict[str, float]:
    """Baseline: species-level stratified shuffle split (can be optimistic due to leakage)."""
    if not HAVE_SKLEARN:
        return {}
    df = species_df.dropna(subset=feature_cols + ["label"]).copy()
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(sss.split(X, y))
    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
        max_features="sqrt",
    )
    clf.fit(X[tr], y[tr])
    yhat = clf.predict(X[te])
    return {
        "accuracy": float(accuracy_score(y[te], yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y[te], yhat)),
        "precision": float(precision_score(y[te], yhat, zero_division=0)),
        "recall": float(recall_score(y[te], yhat, zero_division=0)),
        "f1": float(f1_score(y[te], yhat, zero_division=0)),
    }


def write_markdown_report(
    out_md: Path,
    inv_df: pd.DataFrame,
    species_df: pd.DataFrame,
    feature_cols: List[str],
    redundancy: Dict[str, float],
    cv_model: Optional[pd.DataFrame],
    cv_family: Optional[pd.DataFrame],
    baseline: Optional[Dict[str, float]],
) -> None:
    lines: List[str] = []
    lines.append("# ODE Benchmark Audit Report")
    lines.append("")
    lines.append("This report summarizes per-model scale-label distributions (q80/q99 rule), feature-space diversity, and cross-domain validation using group-aware splits (by model and by family).")
    lines.append("")
    lines.append("## Label rule")
    lines.append("")
    lines.append(f"- A species is labeled **stochastic (label=1)** iff: `q80 < {LABEL_THRESHOLD}` **and** `q99 < {LABEL_THRESHOLD}` (quantiles computed over the full deterministic trajectory).")
    lines.append("")
    lines.append("## Inventory summary (per model)")
    lines.append("")
    # Keep essential columns
    inv_cols = [c for c in ["model","family","n_species","label1_count","label0_count","label1_fraction","q99_min","q99_median","q99_max","oscillatory_species_count","error"] if c in inv_df.columns]
    lines.append(inv_df[inv_cols].to_markdown(index=False))
    lines.append("")

    # Global label distribution
    if not species_df.empty:
        total_species = int(len(species_df))
        total_label1 = int(np.sum(species_df["label"].to_numpy(dtype=int) == 1))
        total_label0 = total_species - total_label1
        lines.append("## Global label distribution")
        lines.append("")
        lines.append(f"- Total species: **{total_species}**")
        lines.append(f"- label=1 (stochastic): **{total_label1}** ({100.0*total_label1/max(1,total_species):.1f}%)")
        lines.append(f"- label=0 (deterministic): **{total_label0}** ({100.0*total_label0/max(1,total_species):.1f}%)")
        lines.append("")

        # Family counts
        fam_counts = species_df.groupby("family")["label"].agg(
            n_species="count",
            label1=lambda s: int(np.sum(s.astype(int)==1)),
            label0=lambda s: int(np.sum(s.astype(int)==0)),
        ).reset_index()
        lines.append("## Per-family label counts")
        lines.append("")
        lines.append(fam_counts.to_markdown(index=False))
        lines.append("")

    # Redundancy
    lines.append("## Feature-space redundancy diagnostic")
    lines.append("")
    if redundancy:
        lines.append(f"- Nearest-neighbor distance (standardized feature space): min={redundancy['nn_min']:.4f}, p05={redundancy['nn_p05']:.4f}, median={redundancy['nn_median']:.4f}")
        lines.append("  - Very small values (e.g. < 0.05) suggest near-duplicate species feature vectors.")
    else:
        lines.append("- (Not computed)")
    lines.append("")

    # CV sections
    if cv_model is not None:
        lines.append("## Cross-domain validation: grouped by **model**")
        lines.append("")
        lines.append(cv_model.to_markdown(index=False))
        lines.append("")
        agg = cv_model[["accuracy","balanced_accuracy","precision","recall","f1","mcc"]].agg(["mean","std"]).T.reset_index()
        agg.columns = ["metric","mean","std"]
        lines.append("**Aggregate across folds:**")
        lines.append("")
        lines.append(agg.to_markdown(index=False))
        lines.append("")

    if cv_family is not None:
        lines.append("## Cross-domain validation: grouped by **family** (leave-one-family-out where possible)")
        lines.append("")
        lines.append(cv_family.to_markdown(index=False))
        lines.append("")
        agg = cv_family[["accuracy","balanced_accuracy","precision","recall","f1","mcc"]].agg(["mean","std"]).T.reset_index()
        agg.columns = ["metric","mean","std"]
        lines.append("**Aggregate across folds:**")
        lines.append("")
        lines.append(agg.to_markdown(index=False))
        lines.append("")

    if baseline is not None and baseline:
        lines.append("## Baseline (optimistic): species-level random 25% split")
        lines.append("")
        lines.append(pd.DataFrame([baseline]).to_markdown(index=False))
        lines.append("")
        lines.append("Note: this baseline can be overly optimistic because it may mix species from the same model/family across train/test.")
        lines.append("")

    # Feature list
    lines.append("## Training feature columns used for validation")
    lines.append("")
    for c in feature_cols:
        lines.append(f"- `{c}`")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", type=str, default=".", help="Directory containing model *.py files.")
    ap.add_argument("--t_final", type=float, default=DEFAULT_T_FINAL, help="Final simulation time (minutes).")
    ap.add_argument("--n_points", type=int, default=DEFAULT_N_POINTS, help="Number of time points for t_eval.")
    ap.add_argument("--out_dir", type=str, default=".", help="Output directory.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for RFC.")
    args = ap.parse_args()

    models_dir = Path(args.models_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = set(DEFAULT_EXCLUDE)
    
    exclude.add(Path(__file__).name)

    print(f"[audit] Scanning models in: {models_dir}")
    species_df, inv_df = build_dataset(models_dir, t_final=args.t_final, n_points=args.n_points, exclude=exclude)

    # Save raw outputs
    inv_csv = out_dir / "audit_inventory.csv"
    sp_csv = out_dir / "audit_species.csv"
    inv_df.to_csv(inv_csv, index=False)
    species_df.to_csv(sp_csv, index=False)
    print(f"[audit] Wrote {inv_csv}")
    print(f"[audit] Wrote {sp_csv}")

    # Validation feature set 
    feature_cols = [
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

    redundancy = {}
    cv_model = None
    cv_family = None
    baseline = None

    if not species_df.empty and HAVE_SKLEARN:
        df = species_df.dropna(subset=feature_cols + ["label"]).copy()
        X = df[feature_cols].to_numpy(dtype=float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        redundancy = compute_redundancy(Xs)

        # Cross-domain validation
        try:
            cv_model = group_cv(species_df, group_col="model", feature_cols=feature_cols, seed=args.seed)
            cv_model.to_csv(out_dir / "cv_by_model.csv", index=False)
            print(f"[audit] Wrote {out_dir / 'cv_by_model.csv'}")
        except Exception as e:
            print(f"[audit] Model-group CV skipped: {e}")

        try:
            cv_family = group_cv(species_df, group_col="family", feature_cols=feature_cols, seed=args.seed)
            cv_family.to_csv(out_dir / "cv_by_family.csv", index=False)
            print(f"[audit] Wrote {out_dir / 'cv_by_family.csv'}")
        except Exception as e:
            print(f"[audit] Family-group CV skipped: {e}")

        # Baseline random split
        try:
            baseline = random_split_baseline(species_df, feature_cols=feature_cols, seed=args.seed)
        except Exception as e:
            print(f"[audit] Baseline split skipped: {e}")

        # PCA plot (optional)
        if HAVE_PLT:
            try:
                pca = PCA(n_components=2, random_state=args.seed)
                Z = pca.fit_transform(Xs)
                fig = plt.figure()
                # Color by family (categorical)
                fam = df["family"].astype(str).to_numpy()
                fam_ids = pd.Categorical(fam).codes
                plt.scatter(Z[:,0], Z[:,1], c=fam_ids, s=18)
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.title("PCA of species features (colored by family)")
                plt.tight_layout()
                fig.savefig(out_dir / "pca_by_family.png", dpi=200)
                plt.close(fig)

                fig = plt.figure()
                lab = df["label"].to_numpy(dtype=int)
                plt.scatter(Z[:,0], Z[:,1], c=lab, s=18)
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.title("PCA of species features (colored by label)")
                plt.tight_layout()
                fig.savefig(out_dir / "pca_by_label.png", dpi=200)
                plt.close(fig)

                print(f"[audit] Wrote PCA plots to {out_dir}")
            except Exception as e:
                print(f"[audit] PCA plot skipped: {e}")

    # Markdown report
    out_md = out_dir / "audit_report.md"
    write_markdown_report(
        out_md=out_md,
        inv_df=inv_df,
        species_df=species_df,
        feature_cols=feature_cols,
        redundancy=redundancy,
        cv_model=cv_model,
        cv_family=cv_family,
        baseline=baseline,
    )
    print(f"[audit] Wrote {out_md}")


if __name__ == "__main__":
    main()
