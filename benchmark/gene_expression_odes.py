# gene_expression_numeric.py
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


# (opcional) solver_options si quieres
solver_options = {
    "method": "Radau",
    "rtol": 1e-6,
    "atol": 1e-9,
    "max_step": 0.5,
}