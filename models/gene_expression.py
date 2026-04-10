# =============================================================================
# gene_expression.py
# -----------------------------------------------------------------------------
# Minimal gene expression benchmark model used in HySimODE.
#
# This script defines a simple two-species system representing mRNA (M) and
# protein (P) dynamics. The model includes:
#   - constitutive transcription of mRNA at rate α
#   - first-order degradation of mRNA at rate δ_m
#   - translation of protein from mRNA at rate β
#   - first-order degradation of protein at rate δ_p
#
# The system is linear and does not include regulatory feedback, making it a
# useful baseline for evaluating the behavior of the hybrid stochastic–
# deterministic simulation framework under well-understood dynamics.
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


solver_options = {
    "method": "Radau",
    "rtol": 1e-6,
    "atol": 1e-9,
    "max_step": 0.5,
}