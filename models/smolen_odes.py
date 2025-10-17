# =============================================================================
#  smolen_odes.py
# -----------------------------------------------------------------------------
#  Description:
#      ODE model (Smolen et al.) in concentration units (µM) for synaptic
#      tagging and capture with pulsed kinase activation. Exposes a robust
#      stimulus function with configurable rectangular or smooth-edged pulses.
#
#  Units and compartments:
#      - State variables and ODEs are expressed in µM.
#      - Two compartments with different volumes:
#            synapse  (vol_syn = 0.2 fL)
#            dendrite (vol_dend = 2.0 fL)
#        Indices are provided via 'syn_indices' and 'dend_indices'.
#
#  Stimulus:
#      - Times (min): 'stim_times'
#      - Basal and peak Ca2+ levels: 'Cabas', 'AMPTETCA', 'AMPTETCAD'
#      - Robust configuration:
#            USE_ROBUST_STIM : True  -> boxcar pulses (rectangular or smooth tanh edges)
#            PULSE_WIDTH     : duration (min)
#            SMOOTH_EDGES    : use tanh edges if True
#            EDGE_WIDTH      : edge width for tanh smoothing
#
#  Exports (required by HySimODE-compatible tooling):
#      - params : dict of model parameters (in consistent units)
#      - Y0     : initial conditions (list/array, in µM)
#      - var_names : list of species names, len(var_names) == len(Y0)
#      - odes(t, y, params) : RHS of the system (returns dy/dt in µM/min)
#      - syn_indices, dend_indices : species-to-compartment mapping
#      - solver_options : recommended options for scipy.integrate.solve_ivp
#
#  Usage:
#      (A) Hybrid simulation via adapter (concentration → molecules):
#          BASE_MODEL=smolen_odes python hysimode.py \
#              --model models/concentration_adapter_hybrid.py --tfinal 500 --dt 0.5 --runs 1
#
#      (B) Feature generation for RFC testing:
#          python make_features_smolen.py
#          # Note: the script internally converts µM → molecules before computing features.
#
#  Notes:
#      - The RFC used in HySimODE was trained on molecule counts. During hybrid
#        simulation, use the concentration adapter to convert units on-the-fly.
#      - For feature generation in this work, conversion is handled inside
#        'make_features_smolen.py' (the adapter is not used for that step).
#
#  © 2025 Criseida G. Zamora Chimal
# =============================================================================

import numpy as np

# === Physical constants ===
NA = 6.022e23
vol_syn = 0.2e-15   # 0.2 fL
vol_dend = 2.0e-15  # 2.0 fL

# === Stimulus parameters ===
stim_times = [100, 105, 110]
Cabas = 0.04
AMPTETCA = 1.4
AMPTETCAD = 0.65

# === Robust stimulus configuration ===
USE_ROBUST_STIM = True     # if False, use simple |t - stim| < 0.05 logic
PULSE_WIDTH = 0.10         # min (duration of boxcar pulse)
SMOOTH_EDGES = False       # if True, use smooth tanh edges
EDGE_WIDTH = 0.02          # smooth edge width (min)


def _level_rect(t):
    """Rectangular pulse for given stim_times."""
    for s in stim_times:
        if s <= t < s + PULSE_WIDTH:
            return 1.0
    return 0.0


def _level_smooth(t):
    """Smooth tanh-edged pulse for given stim_times."""
    lev = 0.0
    for s in stim_times:
        a = 0.5 * (np.tanh((t - s) / EDGE_WIDTH) + 1.0)
        b = 0.5 * (np.tanh((t - (s + PULSE_WIDTH)) / EDGE_WIDTH) + 1.0)
        lev = max(lev, a - b)
    return lev


def stimulus(t):
    """
    Calcium stimulus function.
    Uses either robust boxcar or short rectangular pulses.
    """
    if not stim_times:
        return Cabas, Cabas
    if USE_ROBUST_STIM:
        level = _level_smooth(t) if SMOOTH_EDGES else _level_rect(t)
        casyn = Cabas + (AMPTETCA - Cabas) * level
        cadend = Cabas + (AMPTETCAD - Cabas) * level
    else:
        # original narrow pulse condition
        casyn, cadend = Cabas, Cabas
        if any(abs(t - stim) < 0.05 for stim in stim_times):
            casyn, cadend = AMPTETCA, AMPTETCAD
    return casyn, cadend


# === Model parameters ===
params = {
    "kfck2": 200.0, "tauck2": 1.0, "Kck2": 1.4,
    "kfpp": 2.0, "taupp": 2.0, "Kpp": 0.225,
    "kphos1": 0.45, "kdeph1": 0.006,
    "kphos4": 2.0, "kdeph4": 0.011,
    "kphos5": 0.011, "kdeph5": 0.04,
    "kphos7": 4.0, "kdeph7": 0.1,
    "kphos8": 0.015, "kdeph8": 0.02,
    "ktrans1": 1.2, "vtrbas1": 0.00005, "tauprp": 45.0,
    "vtrbaspkm": 0.0003, "taupkm": 50.0,
    "kpkmon": 0.5, "kpkmon2": 0.055, "Kpkm": 0.75,
    "kexpkm": 0.0025, "klkpkm": 0.012, "Vsd": 0.03,
    "kltp": 0.014, "tfsyn": 30.0, "vfbas": 0.01,
    "kltd": 0.03, "tnsyn": 600.0, "vdbas": 0.0033,
    "kfbasraf": 0.0001, "kbraf": 0.12,
    "kfmkk": 0.6, "kbmkk": 0.025, "Kmkk": 0.25,
    "kferk": 0.52, "kberk": 0.025, "Kmk": 0.25,
    "raftot": 0.25,
    "mkktot": 0.25,
    "erktot": 0.25,
}

# === ODEs ===
def odes(t, y, params):
    (CaMKII, Raf_s, MEK_s, MEKpp_s, ERK_s, ERKpp_s, PP, tagp1,
     tagd1, tagd2, CaMK_d, Raf_d, MEK_d, MEKpp_d, ERK_d, ERKpp_d,
     psit1, psit2, PRP, PKM_d, PKM_s, fsyn, nsyn) = y

    casyn, cadend = stimulus(t)

    powca = casyn**4
    powcad = cadend**4
    powkc = params["Kck2"]**4
    powkc2 = params["Kpp"]**4
    powkcd = 0.6**4

    Rafp_s = params["raftot"] - Raf_s
    MEKp_s = params["mkktot"] - MEK_s - MEKpp_s
    ERKp_s = params["erktot"] - ERK_s - ERKpp_s
    Rafp_d = params["raftot"] - Raf_d
    MEKp_d = params["mkktot"] - MEK_d - MEKpp_d
    ERKp_d = params["erktot"] - ERK_d - ERKpp_d

    dCaMKII = params["kfck2"] * (powca / (powca + powkc)) - CaMKII / params["tauck2"]
    dRaf_s = -params["kfbasraf"] * Raf_s + params["kbraf"] * Rafp_s
    dMEK_s = -params["kfmkk"] * Rafp_s * MEK_s / (MEK_s + params["Kmkk"]) + params["kbmkk"] * MEKp_s / (MEKp_s + params["Kmkk"])
    dMEKpp_s = params["kfmkk"] * Rafp_s * MEKp_s / (MEKp_s + params["Kmkk"]) - params["kbmkk"] * MEKpp_s / (MEKpp_s + params["Kmkk"])
    dERK_s = -params["kferk"] * MEKpp_s * ERK_s / (ERK_s + params["Kmk"]) + params["kberk"] * ERKp_s / (ERKp_s + params["Kmk"])
    dERKpp_s = params["kferk"] * MEKpp_s * ERKp_s / (ERKp_s + params["Kmk"]) - params["kberk"] * ERKpp_s / (ERKpp_s + params["Kmk"])
    ERKact_s = ERKpp_s
    dPP = params["kfpp"] * (powca / (powca + powkc2)) - PP / params["taupp"]
    dtagp1 = params["kphos1"] * CaMKII * (1 - tagp1) - params["kdeph1"] * tagp1
    dtagd1 = params["kphos4"] * ERKact_s * (1 - tagd1) - params["kdeph4"] * tagd1
    dtagd2 = params["kdeph5"] * PP * (1 - tagd2) - params["kphos5"] * tagd2

    TLTP = tagp1**2
    TLTD = tagd1 * tagd2

    dCaMK_d = params["kfck2"] * (powcad / (powcad + powkcd)) - CaMK_d / params["tauck2"]
    dRaf_d = -params["kfbasraf"] * Raf_d + params["kbraf"] * Rafp_d
    dMEK_d = -params["kfmkk"] * Rafp_d * MEK_d / (MEK_d + params["Kmkk"]) + params["kbmkk"] * MEKp_d / (MEKp_d + params["Kmkk"])
    dMEKpp_d = params["kfmkk"] * MEKp_d * MEK_d / (MEK_d + params["Kmkk"]) - params["kbmkk"] * MEKpp_d / (MEKpp_d + params["Kmkk"])
    dERK_d = -params["kferk"] * MEKpp_d * ERK_d / (ERK_d + params["Kmk"]) + params["kberk"] * ERKp_d / (ERKp_d + params["Kmk"])
    dERKpp_d = params["kferk"] * MEKpp_d * ERKp_d / (ERKp_d + params["Kmk"]) - params["kberk"] * ERKpp_d / (ERKpp_d + params["Kmk"])
    ERKact_d = ERKpp_d
    dpsit1 = params["kphos7"] * ERKact_d * (1 - psit1) - params["kdeph7"] * psit1
    dpsit2 = params["kphos8"] * CaMK_d * (1 - psit2) - params["kdeph8"] * psit2
    dPRP = params["ktrans1"] * psit1**2 - PRP / params["tauprp"] + params["vtrbas1"]
    dPKM_d = params["kpkmon"] * psit1 * psit2 - PKM_d / params["taupkm"] + params["vtrbaspkm"] - params["kexpkm"] * TLTP * PKM_d + params["Vsd"] * params["klkpkm"] * PKM_s
    dPKM_s = params["kexpkm"] * TLTP * PKM_d / params["Vsd"] - params["klkpkm"] * PKM_s + params["vtrbaspkm"] + params["kpkmon2"] * PKM_s**2 / (PKM_s**2 + params["Kpkm"]**2) - PKM_s / params["taupkm"]
    dfsyn = params["kltp"] * PKM_s - fsyn / params["tfsyn"] + params["vfbas"]
    dnsyn = -params["kltd"] * TLTD * PRP * nsyn - nsyn / params["tnsyn"] + params["vdbas"]

    return [
        dCaMKII, dRaf_s, dMEK_s, dMEKpp_s, dERK_s, dERKpp_s, dPP, dtagp1,
        dtagd1, dtagd2, dCaMK_d, dRaf_d, dMEK_d, dMEKpp_d, dERK_d, dERKpp_d,
        dpsit1, dpsit2, dPRP, dPKM_d, dPKM_s, dfsyn, dnsyn
    ]

# === Initial conditions and variable names ===
Y0 = [0.001, 0.125, 0.075, 0.1, 0.075, 0.1, 0.001, 0.001, 0.001, 0.001,
      0.001, 0.125, 0.075, 0.1, 0.075, 0.1, 0.001, 0.001, 0.001, 0.001,
      0.01, 0.01, 0.01]

var_names = [
    "CaMKII", "Raf_s", "MEK_s", "MEKpp_s", "ERK_s", "ERKpp_s", "PP", "tagp1",
    "tagd1", "tagd2", "CaMK_d", "Raf_d", "MEK_d", "MEKpp_d", "ERK_d", "ERKpp_d",
    "psit1", "psit2", "PRP", "PKM_d", "PKM_s", "fsyn", "nsyn"
]

# === Compartments ===
syn_indices = list(range(10)) + [20, 21, 22]
dend_indices = list(range(10, 20))

# === Recommended solver options ===
solver_options = {
    "method": "BDF",
    "rtol": 1e-6,
    "atol": 1e-9,
    "max_step": 0.02
}
