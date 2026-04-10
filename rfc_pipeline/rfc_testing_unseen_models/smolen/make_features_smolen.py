# -----------------------------------------------------------------------------
#  make_features_smolen.py
# -----------------------------------------------------------------------------
#  Description:
#      This script computes quantitative trajectory features for each species
#      in a concentration-based biochemical ODE model using the HySimODE
#      simulation conventions. It integrates the system deterministically and
#      extracts the same feature set used during Random Forest Classifier (RFC)
#      training, enabling consistent evaluation on unseen models.
#
#      The script supports models defined via an adapter (e.g.,
#      concentration_adapter_hybrid.py), ensuring compatibility with
#      concentration-to-molecule transformations and HySimODE's internal
#      simulation pipeline.
#
#  Features computed per species:
#      Robust windowed and global features:
#          initial_q50, initial_cv, initial_minmax_ratio, initial_nmadydt,
#          final_q50, final_cv, final_minmax_ratio, final_nmadydt,
#          cv_total, minmax_ratio_total, nmadydt_total
#
#      These features capture:
#          - typical abundance (median-based statistics)
#          - relative variability (coefficient of variation)
#          - oscillatory amplitude / transients (min–max ratio)
#          - normalized temporal variation (derivative-based metrics)
#
#  Methodology:
#      - The ODE system is integrated using solve_ivp with solver settings
#        consistent with HySimODE (fixed max_step = dt).
#      - Feature extraction is performed over:
#            • an initial transient window
#            • a final observation window
#            • the full trajectory
#      - All features are numerically stabilized using small epsilon values
#        and sanitized to avoid NaN or infinite outputs.
#
#  Output:
#      - A CSV file (e.g., features_smolen.csv) containing one row per species
#        with all computed features, along with species indices and names.
#
#  Notes:
#      - The feature definitions are consistent with those used during RFC
#        training, ensuring compatibility for downstream classification.
#      - The BASE_MODEL environment variable is used to dynamically select
#        the underlying biochemical model when using an adapter.
#      - This script is intended for evaluation on unseen models and does not
#        perform classification directly.
#
#  © 2025 Criseida G. Zamora Chimal.
# -----------------------------------------------------------------------------

import os, argparse, importlib
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

OBS_WINDOW = 100.0
INITIAL_WINDOW = 50.0
EPS = 1e-12

def compute_features_exact(t, Y):
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)
    nvars = Y.shape[0]
    t_end = float(t[-1])
    mask_final = t >= (t_end - OBS_WINDOW)
    mask_initial = t <= INITIAL_WINDOW

    rows = []
    for i in range(nvars):
        y = np.maximum(Y[i, :], 0.0)
        dy = np.gradient(y, t)

        mean_total = float(np.mean(y))
        std_total = float(np.std(y))
        cv_total = float(std_total / (mean_total + EPS))
        madydt_total = float(np.mean(np.abs(dy)))

        ymin, ymax = float(np.min(y)), float(np.max(y))
        minmax_ratio_total = float((ymin + EPS) / (ymax + EPS))

        yi = y[mask_initial]
        yf = y[mask_final]

        initial_q50 = float(np.quantile(yi, 0.50)) if yi.size else 0.0
        final_q50   = float(np.quantile(yf, 0.50)) if yf.size else 0.0

        initial_cv = float(np.std(yi) / (float(np.mean(yi)) + EPS)) if yi.size else 0.0
        final_cv   = float(np.std(yf) / (float(np.mean(yf)) + EPS)) if yf.size else 0.0

        initial_minmax_ratio = float((float(np.min(yi)) + EPS) / (float(np.max(yi)) + EPS)) if yi.size else 0.0
        final_minmax_ratio   = float((float(np.min(yf)) + EPS) / (float(np.max(yf)) + EPS)) if yf.size else 0.0

        initial_nmadydt = float(np.mean(np.abs(dy[mask_initial])) / (initial_q50 + EPS)) if yi.size else 0.0
        final_nmadydt   = float(np.mean(np.abs(dy[mask_final]))   / (final_q50 + EPS))   if yf.size else 0.0

        median_total = float(np.quantile(y, 0.50))
        nmadydt_total = float(madydt_total / (median_total + EPS))

        row = dict(
            initial_q50=initial_q50,
            initial_cv=initial_cv,
            initial_minmax_ratio=initial_minmax_ratio,
            initial_nmadydt=initial_nmadydt,
            final_q50=final_q50,
            final_cv=final_cv,
            final_minmax_ratio=final_minmax_ratio,
            final_nmadydt=final_nmadydt,
            cv_total=cv_total,
            minmax_ratio_total=minmax_ratio_total,
            nmadydt_total=nmadydt_total,
        )
        # sanitize
        for k,v in row.items():
            if not np.isfinite(v):
                row[k] = 0.0
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="smolen_odes")
    ap.add_argument("--adapter", default="concentration_adapter_hybrid")
    ap.add_argument("--tfinal", type=float, default=460.0)
    ap.add_argument("--dt", type=float, default=0.01)  
    ap.add_argument("--out", default="features_smolen.csv")
    args = ap.parse_args()

    os.environ["BASE_MODEL"] = args.base_model
    A = importlib.import_module(args.adapter)
    A = importlib.reload(A)

    dt = float(args.dt)
    t_eval = np.linspace(0.0, args.tfinal, int(args.tfinal / dt) + 1)

    solver_options = getattr(A, "solver_options", {
        "method": "BDF",
        "rtol": 1e-6,
        "atol": 1e-9,
        "max_step": dt
    })
    solver_options = dict(solver_options)
    solver_options["max_step"] = dt 

    sol = solve_ivp(
        lambda t, y: A.odes(t, y, A.params),
        [0.0, args.tfinal],
        np.array(A.Y0, dtype=float),
        t_eval=t_eval,
        **solver_options
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    Y_det = np.maximum(sol.y, 0.0)
    feats = compute_features_exact(sol.t, Y_det)

    out_df = feats.copy()
    out_df.insert(0, "species_name", list(A.var_names))
    out_df.insert(0, "species_index", np.arange(len(A.var_names), dtype=int))
    out_df.to_csv(args.out, index=False)
    print(f"[INFO] Saved: {args.out}")

if __name__ == "__main__":
    main()