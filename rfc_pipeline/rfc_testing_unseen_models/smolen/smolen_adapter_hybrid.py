# smolen_adapter_hybrid.py
# Use smolen_odes.py as source to convert concentration to molecules
# Use a robust stimulues (soft boxcar suave) ~0.1 min.

import numpy as np
import math
import importlib

# --- Load model
S = importlib.import_module("smolen_odes")

# --- Constants and volumes
NA       = 6.022e23
VOL_SYN  = getattr(S, "vol_syn",  2.0e-16)  # L
VOL_DEND = getattr(S, "vol_dend", 2.0e-15)  # L

# --- Nnames and states
var_names = list(getattr(S, "var_names"))
Y0_uM     = np.array(S.Y0, dtype=float)
nvars     = len(Y0_uM)
assert len(var_names) == nvars, "var_names should match with states"

# --- Map of volumes - assign per specie
syn_indices  = list(range(10)) + [20,21,22]   # synapsis: 0..9, PKM_s, fsyn, nsyn
dend_indices = list(range(10,20))             # dendrite: 10..19

def build_V_vector(n):
    V = np.empty(n, dtype=float)
    for i in range(n):
        V[i] = VOL_SYN if i in syn_indices else VOL_DEND
    return V

V    = build_V_vector(nvars)
Vcol = V.reshape(-1,1)

# --- Conversion µM <-> moléculas
def conc_to_mol(c_uM):
    # c[µM] * 1e-6 * V[L] * NA = molecules
    if c_uM.ndim == 1:
        return c_uM * 1e-6 * V * NA
    return c_uM * 1e-6 * Vcol * NA

def mol_to_conc(n_mol):
    # n / (V*NA) * 1e6 = µM
    if n_mol.ndim == 1:
        return (n_mol / (V * NA)) * 1e6
    return (n_mol / (Vcol * NA)) * 1e6

def dconc_to_dmol(dc_uM_dt):
    if dc_uM_dt.ndim == 1:
        return dc_uM_dt * 1e-6 * V * NA
    return dc_uM_dt * 1e-6 * Vcol * NA

# --- Robust stimulues (~0.1 min total)
#     Option A: abs(t-stim)<0.05
PULSE_WIDTH = 0.1   # minutes
Cabas       = getattr(S, "Cabas", 0.04)
AMPTETCA    = getattr(S, "AMPTETCA", 1.4)
AMPTETCAD   = getattr(S, "AMPTETCAD", 0.65)  # tu valor

stim_times  = list(getattr(S, "stim_times", []))

USE_SMOOTH_BOXCAR = False  # pon True if prefer soft edges
EPS_EDGE = 0.02            # width

def _level_rect(t):
    # 1 si t ∈ [s, s+PULSE_WIDTH), 0 
    for s in stim_times:
        if s <= t < s + PULSE_WIDTH:
            return 1.0
    return 0.0

def _level_smooth(t):
    # Max of boxcars for all pulses
    lev = 0.0
    for s in stim_times:
        a = 0.5*(math.tanh((t - s)/EPS_EDGE) + 1.0)
        b = 0.5*(math.tanh((t - (s + PULSE_WIDTH))/EPS_EDGE) + 1.0)
        lev = max(lev, a - b)
    return lev

def stimulus_robust(t):
    if not stim_times:
        return Cabas, Cabas
    level = _level_smooth(t) if USE_SMOOTH_BOXCAR else _level_rect(t)
    casyn  = Cabas + (AMPTETCA  - Cabas) * level
    cadend = Cabas + (AMPTETCAD - Cabas) * level
    return casyn, cadend


#     closure to call S.smolen_odes 
_original_stimulus = S.stimulus if hasattr(S, "stimulus") else None

def smolen_odes_with_robust_stimulus(t, y_uM):
    # Monkey-patch temporal
    if _original_stimulus is not None:
        S.stimulus = stimulus_robust
    dy = S.smolen_odes(t, y_uM)
    # cleaning
    if _original_stimulus is not None:
        S.stimulus = _original_stimulus
    return dy

# --- ODE - molecules
def smolen_odes_mol(t, y_mol):
    y_uM   = mol_to_conc(np.array(y_mol, dtype=float))
    dy_uM  = np.array(smolen_odes_with_robust_stimulus(t, y_uM), dtype=float)  # µM/min
    dy_mol = dconc_to_dmol(dy_uM)                                              # mol/min
    return dy_mol

# --- Y0 en mol y params
Y0     = conc_to_mol(Y0_uM)
params = getattr(S, "params", {})

# --- Suggestion to solver in hybrid simulator
#     use method='BDF' y max_step <= min(0.02, DT) if pulses are 0.1 min.
