# =============================================================================
# gene_expression.py
# -----------------------------------------------------------------------------
# Minimal gene expression benchmark model with explicit production and
# degradation decomposition used in HySimODE.
#
# This script defines a simple two-species system representing mRNA (M) and
# protein (P) dynamics. The model includes:
#   - constitutive transcription of mRNA at rate α
#   - first-order degradation of mRNA at rate δ_m
#   - translation of protein from mRNA at rate β
#   - first-order degradation of protein at rate δ_p
#
# In addition to the standard ODE formulation, the model provides an
# `odes_prod_deg` function that separates the dynamics into positive
# (production) and negative (degradation) contributions. This representation
# is required by HySimODE for hybrid simulation, where stochastic updates
# depend explicitly on production and degradation fluxes.
#
# The system is linear and does not include regulatory feedback, making it a
# controlled benchmark for evaluating hybrid stochastic–deterministic behavior
# and assessing noise quantification under explicit reaction flux decomposition.
#
# The ODE system is solved using a stiff integrator (Radau) with specified
# tolerances to ensure numerical stability.
#
# © 2026 Criseida G. Zamora Chimal
# =============================================================================
import numpy as np

var_names = ["M", "P"]

Y0 = [0.0, 0.0]

params = {
    "alpha": 5.0,
    "delta_m": 1.0,
    "beta": 50.0,
    "delta_p": 0.1,
}

def odes(t, y, params):
    M, P = y
    alpha   = params["alpha"]
    delta_m = params["delta_m"]
    beta    = params["beta"]
    delta_p = params["delta_p"]

    dM = alpha - delta_m * M
    dP = beta * M - delta_p * P
    return np.array([dM, dP], dtype=float)

def odes_prod_deg(t, y, params):
    M, P = y
    alpha   = params["alpha"]
    delta_m = params["delta_m"]
    beta    = params["beta"]
    delta_p = params["delta_p"]

    # términos positivos (producción)
    prod_M = alpha
    prod_P = beta * M

    # términos negativos (degradación, en valor absoluto)
    deg_M = delta_m * M
    deg_P = delta_p * P

    prod = np.array([prod_M, prod_P], dtype=float)
    deg  = np.array([deg_M, deg_P], dtype=float)
    return prod, deg

# (opcional) solver_options si quieres
solver_options = {
    "method": "Radau",
    "rtol": 1e-6,
    "atol": 1e-9,
    "max_step": 0.5,
}