# =============================================================================
#  concentration_adapter_hybrid.py
# -----------------------------------------------------------------------------
#  Description:
#      Generic adapter that converts concentration-based ODE models (in µM)
#      into molecule-based systems for hybrid deterministic–stochastic
#      simulations within HySimODE.
#
#  Functionality:
#      - Dynamically loads a base concentration-based ODE model
#        (e.g., smolen_odes.py)
#      - Converts between µM and molecule units for all species
#      - Preserves the original model structure and parameterization
#      - Supports multi-compartment models (e.g., synaptic/dendritic volumes)
#      - Exports a wrapped ODE function in molecule units, ready for use by
#        hysimode.py or other hybrid solvers
#
#  Unit conversions:
#      µM  →  molecules :  c_uM * 1e-6 * V * N_A
#      molecules → µM :  (n / (V * N_A)) * 1e6
#
#  Notes:
#      - Automatically detects the base model name either from:
#          (1) Command-line argument, e.g.:
#              python concentration_adapter_hybrid.py smolen_odes
#          (2) Environment variable BASE_MODEL (used by hysimode.py)
#          (3) Defaults to "smolen_odes" if none is provided.
#      - Volume parameters are read from the model if available, otherwise
#        default to 1e-15 L.
#      - The adapter exports:
#            params, Y0, var_names, odes(), solver_options
#        which are fully compatible with HySimODE.
#
#  © 2025 Criseida G. Zamora Chimal
# =============================================================================

import numpy as np
import math
import importlib
import os
import sys

# --- Determine model name robustly ---
# Priority:
#   1. If called directly: python concentration_adapter_hybrid.py model_name
#   2. If called from hysimode.py: uses environment variable BASE_MODEL
#   3. Otherwise defaults to "smolen_odes"
if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    MODEL_NAME = sys.argv[1]
else:
    MODEL_NAME = os.environ.get("BASE_MODEL", "smolen_odes")

print(f"[INFO] Loading base model: {MODEL_NAME}")
M = importlib.import_module(MODEL_NAME)

if hasattr(M, "odes"):
    base_odes = M.odes
else:
    raise AttributeError(
        f"Model '{MODEL_NAME}' must define an ODE function named 'odes(t, y, params)'."
    )

params = getattr(M, "params", {})

# --- Get base production/degradation decomposition, if available ---

base_odes_prod_deg = getattr(M, "odes_prod_deg", None)

params = getattr(M, "params", {})
Y0_uM = np.array(M.Y0, dtype=float)
var_names = list(M.var_names)
nvars = len(Y0_uM)

# === Volumes and compartments ===

NA = 6.022e23

# 1) Fully general: per-species volumes (liters)
if hasattr(M, "volumes_L"):
    volumes = np.asarray(M.volumes_L, dtype=float)
    if volumes.shape != (nvars,):
        raise ValueError(
            f"Model '{MODEL_NAME}': volumes_L must have shape ({nvars},), got {volumes.shape}"
        )

# 2) General multi-compartment interface:
#    compartment_indices: dict[str, list[int]]
#    compartment_volumes_L: dict[str, float]
elif hasattr(M, "compartment_indices") and hasattr(M, "compartment_volumes_L"):
    volumes = np.full(nvars, float(getattr(M, "vol_default", 1e-15)), dtype=float)

    for cname, idx_list in M.compartment_indices.items():
        if cname not in M.compartment_volumes_L:
            raise ValueError(
                f"Model '{MODEL_NAME}': compartment '{cname}' has indices but no volume in compartment_volumes_L"
            )
        V = float(M.compartment_volumes_L[cname])
        for i in idx_list:
            volumes[int(i)] = V


# 4) Simple default: single-compartment volume
else:
    volumes = np.full(nvars, float(getattr(M, "vol_default", 1e-15)), dtype=float)

# Sanity checks
if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
    raise ValueError(
        f"Model '{MODEL_NAME}': invalid volumes (must be finite and > 0)."
    )

# === Conversion utilities ===
def conc_to_mol(c_uM):
    """Convert concentrations (µM) to molecule counts."""
    return c_uM * 1e-6 * volumes * NA

def mol_to_conc(n_mol):
    """Convert molecule counts to concentrations (µM)."""
    return (n_mol / (volumes * NA)) * 1e6

def dconc_to_dmol(dc_uM_dt):
    """Convert derivatives from µM/min to molecules/min."""
    return dc_uM_dt * 1e-6 * volumes * NA

# === Wrapped ODE for molecule-based simulation ===
def odes(t, y_mol, params=params):
    """
    Wrapper ODE system that runs the base model in molecule units.
    Converts y from molecules to µM, calls the model ODE, and converts back.
    """
    y_uM = mol_to_conc(np.array(y_mol, dtype=float))
    dy_uM = np.asarray(base_odes(t, y_uM, params), dtype=float)
    dy_mol = dconc_to_dmol(dy_uM)
    return dy_mol


def odes_prod_deg(t, y_mol, params=params):
    """
    Wrapper for production/degradation decomposition in molecule units.

    If the base model provides odes_prod_deg(t, y, params), use it.
    Otherwise, fallback to a net-drift decomposition using base_odes.
    """
    y_uM = mol_to_conc(np.array(y_mol, dtype=float))

    if base_odes_prod_deg is not None:
        # Preferred: true decomposition in concentration units (µM/min)
        prod_uM, deg_uM = base_odes_prod_deg(t, y_uM, params)
        prod_uM = np.asarray(prod_uM, dtype=float)
        deg_uM  = np.asarray(deg_uM,  dtype=float)

        if prod_uM.shape != deg_uM.shape:
            raise ValueError(
                f"[ERROR] base_odes_prod_deg returned prod and deg with different shapes: "
                f"prod{prod_uM.shape}, deg{deg_uM.shape}"
            )
    else:
        # Fallback: derive decomposition from net drift in concentration units
        dy_uM = np.asarray(base_odes(t, y_uM, params), dtype=float)
        prod_uM = np.maximum(dy_uM, 0.0)
        deg_uM  = np.maximum(-dy_uM, 0.0)

    # Convert µM/min → molecules/min
    prod_mol = dconc_to_dmol(prod_uM)
    deg_mol  = dconc_to_dmol(deg_uM)

    return prod_mol, deg_mol

# === Initial conditions (converted to molecules) ===
Y0 = conc_to_mol(Y0_uM)

# === Exported variables (the simulator will import these) ===
solver_options = getattr(M, "solver_options", {
    "method": "BDF",
    "rtol": 1e-6,
    "atol": 1e-9,
    "max_step": 0.02
})
