import numpy as np

# === Model parameters ===
params = {
    "k1": 0.1,   # µM⁻¹·min⁻¹
    "k2": 0.05,  # min⁻¹
    "k3": 0.01   # min⁻¹
}

# === Variables ===
# y = [S, E, SE]
# S = substrate, E = enzyme, SE = substrate-enzyme complex

var_names = ["S", "E", "SE"]

# === Initial conditions (µM) ===
Y0 = [10.0, 1.0, 0.0]

# === Volumes (liters) ===
# This model only has one compartment, so we define a single volume.
vol_cell = 1e-15  # 1 fL typical small cell
volumes = {"cell": vol_cell}

# === ODEs in µM/min ===
def odes(t, y, params):
    S, E, SE = y

    # Binding
    v1 = params["k1"] * S * E
    # Dissociation
    v2 = params["k2"] * SE
    # Product formation (enzyme release)
    v3 = params["k3"] * SE

    dS  = -v1 + v2
    dE  = -v1 + v2 + v3
    dSE = v1 - v2 - v3

    return [dS, dE, dSE]
