# concentration_adapter_hybrid.py
# Generic adapter: models defined in concentrations (µM) → ODEs in molecule counts.
# Optionally overrides the model's stimulus with a robust boxcar to preserve pulse area.

import numpy as np
import math
import importlib

# === Load the original model module (must be passed via --model in hysimode.py) ===
# We assume hysimode.py will import this file as a module; so here we import
# the actual model the user wants to adapt. Replace "MODEL_MODULE_NAME" at runtime:
# hysimode.py will import this adapter directly, so we dynamically import the
# sibling module name from an attribute if provided.
#
# USAGE: you do NOT run this file directly. You run:
#   python hysimode.py --model concentration_adapter_hybrid.py ...
#
# And within this adapter, set MODEL_NAME below to the actual model file stem,
# e.g. "smolen_odes" (without .py). To keep it general, we allow an env var.
import os

MODEL_NAME = os.environ.get("HYSIMODE_BASE_MODEL", "smolen_odes")
M = importlib.import_module(MODEL_NAME)

# ==== Required symbols from the base model (with fallbacks) ====
var_names = list(getattr(M, "var_names"))
Y0_uM     = np.array(getattr(M, "Y0"), dtype=float)
params    = getattr(M, "params", {})

assert len(var_names) == len(Y0_uM), "var_names must match length of Y0"

# Volumes (L). The base model should export them; otherwise set sensible defaults.
vol_syn  = getattr(M, "vol_syn",  2.0e-16)  # 0.2 fL
vol_dend = getattr(M, "vol_dend", 2.0e-15)  # 2.0 fL
NA = 6.022e23

# Compartment mapping. Prefer lists exported by the model; otherwise use Smolen's shape.
syn_indices  = list(getattr(M, "syn_indices",  list(range(10)) + [20, 21, 22]))
dend_indices = list(getattr(M, "dend_indices", list(range(10, 20))))

def _build_volume_vector(n):
    V = np.empty(n, dtype=float)
    syn_set  = set(syn_indices)
    dend_set = set(dend_indices)
    for i in range(n):
        if i in syn_set:
            V[i] = vol_syn
        elif i in dend_set:
            V[i] = vol_dend
        else:
            # If the model defines extra compartments, extend this logic:
            # default to dendrite-sized volume as a conservative choice.
            V[i] = vol_dend
    return V

V = _build_volume_vector(len(Y0_uM))
Vcol = V.reshape(-1, 1)

# === µM ↔ molecules conversions ===
def conc_to_mol(c_uM):
    # c[µM] * 1e-6 * V[L] * NA = molecules
    c = np.asarray(c_uM, dtype=float)
    if c.ndim == 1:
        return c * 1e-6 * V * NA
    return c * 1e-6 * Vcol * NA

def mol_to_conc(n_mol):
    # n / (V*NA) * 1e6 = µM
    n = np.asarray(n_mol, dtype=float)
    if n.ndim == 1:
        return (n / (V * NA)) * 1e6
    return (n / (Vcol * NA)) * 1e6

def dconc_to_dmol(dc_uM_dt):
    d = np.asarray(dc_uM_dt, dtype=float)
    if d.ndim == 1:
        return d * 1e-6 * V * NA
    return d * 1e-6 * Vcol * NA

# === Optional robust stimulus override (boxcar) ===
# If the base model defines a time-dependent stimulus(), adaptive solvers can under-sample short pulses.
# To preserve effective pulse area, you can force a rectangular (or smoothed) boxcar of fixed width.
FORCE_ROBUST_STIMULUS = True  # set False to use the model's native stimulus as-is
PULSE_WIDTH = 0.10            # minutes; match your original effective width
SMOOTH_EDGES = False          # if True, use smooth tanh edges (same area approx.)
EDGE_WIDTH = 0.02             # smoothing width for tanh edges

# Pull base stimulus settings if present
stim_times  = list(getattr(M, "stim_times", []))
Cabas       = getattr(M, "Cabas", 0.04)
AMPTETCA    = getattr(M, "AMPTETCA", 1.4)
AMPTETCAD   = getattr(M, "AMPTETCAD", 0.65)

def _level_rect(t):
    for s in stim_times:
        if s <= t < s + PULSE_WIDTH:
            return 1.0
    return 0.0

def _level_smooth(t):
    lev = 0.0
    for s in stim_times:
        a = 0.5*(math.tanh((t - s)/EDGE_WIDTH) + 1.0)
        b = 0.5*(math.tanh((t - (s + PULSE_WIDTH))/EDGE_WIDTH) + 1.0)
        lev = max(lev, a - b)
    return lev

def _stimulus_boxcar(t):
    if not stim_times:
        return Cabas, Cabas
    level = _level_smooth(t) if SMOOTH_EDGES else _level_rect(t)
    casyn  = Cabas + (AMPTETCA  - Cabas) * level
    cadend = Cabas + (AMPTETCAD - Cabas) * level
    return casyn, cadend

# Keep a handle to the model's original stimulus if it exists
_model_has_stimulus = hasattr(M, "stimulus")

def _rhs_in_uM(t, y_uM):
    """Call the base model RHS in µM/min, optionally overriding stimulus with a robust boxcar."""
    if hasattr(M, "smolen_odes"):
        base_rhs = M.smolen_odes
    elif hasattr(M, "odes"):
        base_rhs = M.odes
    else:
        raise AttributeError("Base model must define `smolen_odes` or `odes`.")

    if FORCE_ROBUST_STIMULUS and _model_has_stimulus:
        # Monkey patch stimulus during this call
        orig = M.stimulus
        M.stimulus = _stimulus_boxcar
        try:
            dy = base_rhs(t, y_uM, params) if base_rhs.__code__.co_argcount >= 3 else base_rhs(t, y_uM)
        finally:
            M.stimulus = orig
        return np.asarray(dy, dtype=float)

    # No override: use model as-is
    return np.asarray(base_rhs(t, y_uM, params) if base_rhs.__code__.co_argcount >= 3
                      else base_rhs(t, y_uM), dtype=float)

# === Public ODE for HySimODE (in MOLECULES) ===
def odes(t, y_mol, _params_unused=None):
    """
    Adapter ODE in molecules/min. Converts state to µM, calls model RHS in µM/min,
    and converts back to molecules/min.
    """
    y_uM  = mol_to_conc(np.asarray(y_mol, dtype=float))
    dy_uM = _rhs_in_uM(t, y_uM)         # µM/min
    dy    = dconc_to_dmol(dy_uM)        # molecules/min
    return dy

# === Exports expected by hysimode.py ===
Y0 = conc_to_mol(Y0_uM)   # molecules
# reuse original params; nothing special needed here
# var_names already defined

# Let models suggest solver settings for short pulses if needed
solver_options = getattr(M, "solver_options", {
    "method": "BDF",
    "rtol": 1e-6,
    "atol": 1e-9,
    "max_step": 0.02 if stim_times else np.inf
})

# Use a finer dt for the deterministic pre-simulation that builds features for the RFC
feature_dt = 0.02 if stim_times else 0.1
