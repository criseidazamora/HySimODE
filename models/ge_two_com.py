# =============================================================================
#  ge_two_com.py
# -----------------------------------------------------------------------------
#  Toy model for testing the generality of the concentration adapter.
#
#  Two-compartment gene expression model:
#      nucleus  : transcription
#      cytosol  : translation
#
#  Species
#      mRNA_nuc
#      mRNA_cyt
#      Protein
#
#  Units
#      concentrations in µM
#
# © 2025 Criseida G. Zamora Chimal
# =============================================================================

import numpy as np

# === Species names ===
var_names = [
    "mRNA_nuc",
    "mRNA_cyt",
    "Protein"
]

# === Initial conditions (µM) ===
Y0 = [
    0.0,   # mRNA_nuc
    0.0,   # mRNA_cyt
    0.0    # Protein
]

# === Parameters ===
params = dict(
    k_tx = 0.05,      # transcription
    k_export = 0.1,   # nuclear export
    k_deg_m = 0.02,   # mRNA degradation
    k_tl = 0.2,       # translation
    k_deg_p = 0.005   # protein degradation
)

# === Compartments (general interface) ===

compartment_indices = {
    "nucleus": [0],
    "cytosol": [1,2]
}

compartment_volumes_L = {
    "nucleus": 0.5e-15,
    "cytosol": 2.0e-15
}

# === ODE system ===
def odes(t, y, p):

    m_n, m_c, P = y

    k_tx = p["k_tx"]
    k_export = p["k_export"]
    k_deg_m = p["k_deg_m"]
    k_tl = p["k_tl"]
    k_deg_p = p["k_deg_p"]

    # nucleus
    dm_n = k_tx - k_export*m_n - k_deg_m*m_n

    # cytosol
    dm_c = k_export*m_n - k_deg_m*m_c

    # protein
    dP = k_tl*m_c - k_deg_p*P

    return np.array([dm_n, dm_c, dP])


# === Optional production/degradation decomposition ===
def odes_prod_deg(t, y, p):

    m_n, m_c, P = y

    k_tx = p["k_tx"]
    k_export = p["k_export"]
    k_deg_m = p["k_deg_m"]
    k_tl = p["k_tl"]
    k_deg_p = p["k_deg_p"]

    prod = np.zeros(3)
    deg  = np.zeros(3)

    # mRNA nucleus
    prod[0] = k_tx
    deg[0]  = k_export*m_n + k_deg_m*m_n

    # mRNA cytosol
    prod[1] = k_export*m_n
    deg[1]  = k_deg_m*m_c

    # protein
    prod[2] = k_tl*m_c
    deg[2]  = k_deg_p*P

    return prod, deg


# === Solver options ===
solver_options = {
    "method": "BDF",
    "rtol": 1e-6,
    "atol": 1e-9,
    "max_step": 0.05
}