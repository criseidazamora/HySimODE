# make_features_smolen.py
# Generate features using smolen_odes.py 
import numpy as np
import pandas as pd
import importlib
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt

# ===== Config =====
MODEL_MODULE   = "smolen_odes"          # file: smolen_odes.py
OUT_CSV        = "features_smolen.csv"
METHOD         = "BDF"
RTOL, ATOL     = 1e-6, 1e-8
# Ventanas para features
OBS_WINDOW     = 100.0
INITIAL_WINDOW = 50.0

# ===== Import model =====
S = importlib.import_module(MODEL_MODULE)

# ===== Time imported from the model =====
t_start = getattr(S, "t_start", 0.0)
t_end   = getattr(S, "t_end",   2000.0)
dt      = getattr(S, "dt",      None)

if dt is not None and dt > 0:
    t_eval_default = np.arange(t_start, t_end + 1e-12, dt)
else:
    # fallback if there is not dt defined
    t_eval_default = np.linspace(t_start, t_end, 1200)

print(f"[INFO] Using time grid from model: t_start={t_start}, t_end={t_end}, dt={dt}")

# ===== Constants/ volumes =====
NA        = 6.022e23
VOL_SYN   = getattr(S, "vol_syn",  2.0e-16)  # L synapse
VOL_DEND  = getattr(S, "vol_dend", 2.0e-15)  # L dendrite

# ===== Names/índex (must come from the model) =====
if not hasattr(S, "var_names"):
    raise RuntimeError(
        "smolen_odes.py should expor 'var_names' (species list) in the same orden as the state."
    )
VARS = list(getattr(S, "var_names"))

if not hasattr(S, "Y0"):
    raise RuntimeError("smolen_odes.py must export'Y0' (incitial conditions in µM).")

nvars = len(S.Y0)
if len(VARS) != nvars:
    raise RuntimeError(
        f"len(var_names)={len(VARS)} != len(Y0)={nvars}. "
        "Please fix 'var_names' so that it matches the order and length of the state vector."
    )

# Compartment mapping (adjust if your ordering differs)
syn_indices  = list(range(10)) + [20, 21, 22]
dend_indices = list(range(10, 20))
def build_V_vector(n):
    V = np.empty(n, dtype=float)
    for i in range(n):
        if i in syn_indices:
            V[i] = VOL_SYN
        elif i in dend_indices:
            V[i] = VOL_DEND
        else:
            V[i] = VOL_DEND
    return V

V = build_V_vector(nvars)

# ===== Conversions µM ⇄ molecules =====
def conc_to_mol(c_uM: np.ndarray) -> np.ndarray:
    # c[µM] * 1e-6 [mol/L] * V[L] * NA [molec/mol] = molec
    return c_uM * 1e-6 * V * NA

def mol_to_conc(n_mol: np.ndarray) -> np.ndarray:
    # n[molec] / (V[L] * NA) * 1e6 = µM
    return (n_mol / (V * NA)) * 1e6

def dconc_to_dmol(dc_uM_dt: np.ndarray) -> np.ndarray:
    return dc_uM_dt * 1e-6 * V * NA

# ===== ODE in molecules (wrapper) =====
# IMPORTANT: always call S.smolen_odes (WITHOUT modifying the model's stimulus).
def smolen_odes_mol(t: float, y_mol: np.ndarray) -> np.ndarray:
    y_conc  = mol_to_conc(np.array(y_mol, dtype=float))
    dy_conc = np.array(S.smolen_odes(t, y_conc), dtype=float)  # [µM/min]
    dy_mol  = dconc_to_dmol(dy_conc)                           # [molec/min]
    return dy_mol

# ===== Deterministic simulation - molecules =====
def simulate_in_molecules():
    Y0_mol = conc_to_mol(np.array(S.Y0, dtype=float))  # Y0 to µM -> molecules
    sol = solve_ivp(
        smolen_odes_mol, (t_start, t_end), Y0_mol,
        method=METHOD, rtol=RTOL, atol=ATOL,
        t_eval=t_eval_default,
        max_step=(dt if dt else 1.0)
    )
    if not sol.success:
        raise RuntimeError("solve_ivp failed en make_features_smolen.")
    Y = np.maximum(sol.y, 0.0)
    return sol.t, Y

# ===== Computing features =====
def compute_features(t: np.ndarray, Y: np.ndarray):
    nvars = Y.shape[0]
    t_end_sim = float(t[-1])
    mask_final   = t >= (t_end_sim - OBS_WINDOW)
    mask_initial = t <=  INITIAL_WINDOW

    rows = []
    for i in range(nvars):
        y  = Y[i, :]
        dy = np.gradient(y, t)

        # Whole trajectory
        mean_total    = float(np.mean(y))
        std_total     = float(np.std(y))
        cv_total      = float(std_total / (mean_total + 1e-12))
        madydt_total  = float(np.mean(np.abs(dy)))
        ymin, ymax    = float(np.min(y)), float(np.max(y))
        minmax_ratio  = float((ymin + 1e-12) / (ymax + 1e-12))

        # Audit: time point of maximum
        imax     = int(np.argmax(y))
        t_of_max = float(t[imax])

        # Final window
        yf  = y[mask_final]; dyf = dy[mask_final]
        mean_final    = float(np.mean(yf))
        std_final     = float(np.std(yf))
        cv_final      = float(std_final / (mean_final + 1e-12))
        madydt_final  = float(np.mean(np.abs(dyf)))
        final_value   = float(y[-1])

        # Initial window
        yi           = y[mask_initial]
        mean_initial = float(np.mean(yi))

        rows.append({
            "mean_total": mean_total,
            "std_total": std_total,
            "cv_total": cv_total,
            "madydt_total": madydt_total,
            "min": ymin, "max": ymax, "minmax_ratio": minmax_ratio,
            "final_mean": mean_final, "final_std": std_final,
            "final_cv": cv_final, "final_madydt": madydt_final,
            "final_value": final_value,
            "initial_mean": mean_initial,
        })
    return rows

def main():
    t, Y = simulate_in_molecules()
    feats = compute_features(t, Y)

    df = pd.DataFrame(feats)
    df.insert(0, "species_name", VARS)
    df.insert(0, "model", MODEL_MODULE)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved features → {OUT_CSV}")

if __name__ == "__main__":
    main()
