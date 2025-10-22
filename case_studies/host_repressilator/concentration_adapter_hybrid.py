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

# === Extract model components ===
if hasattr(M, "odes"):
    base_odes = M.odes
else:
    raise AttributeError(
        f"Model '{MODEL_NAME}' must define an ODE function named 'odes(t, y, params)'."
    )

params = getattr(M, "params", {})
Y0_uM = np.array(M.Y0, dtype=float)
var_names = list(M.var_names)
nvars = len(Y0_uM)

# === Volumes and compartments ===
NA = 6.022e23

# Default single-compartment case
volumes = np.ones(nvars) * getattr(M, "vol_syn", 1e-15)

# If the model defines compartment indices, use them
if hasattr(M, "syn_indices") and hasattr(M, "dend_indices"):
    syn_indices = M.syn_indices
    dend_indices = M.dend_indices
    vol_syn = getattr(M, "vol_syn", 2e-16)
    vol_dend = getattr(M, "vol_dend", 2e-15)
    for i in syn_indices:
        volumes[i] = vol_syn
    for i in dend_indices:
        volumes[i] = vol_dend

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

# === Initial conditions (converted to molecules) ===
Y0 = conc_to_mol(Y0_uM)

# === Exported variables (the simulator will import these) ===
solver_options = getattr(M, "solver_options", {
    "method": "BDF",
    "rtol": 1e-6,
    "atol": 1e-9,
    "max_step": 0.02
})
