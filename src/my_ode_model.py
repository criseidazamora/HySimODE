# =============================================================================
#  my_ode_model.py
# -----------------------------------------------------------------------------
#  Description:
#      Minimal working example of an ODE model compatible with HySimODE.
#      This toy system contains a stochastic-like birth–death species (X)
#      and a deterministic accumulator (Y).
#
#  System:
#      dX/dt = k_prod - k_deg * X       (X may be stochastic)
#      dY/dt = alpha * X - beta * Y     (Y is deterministic)
#
#  Usage example:
#      python hysimode.py --model toy_birthdeath_model.py --tfinal 50 --dt 1
#
#  Required model structure for HySimODE:
#      - A dictionary of parameters (e.g., 'params')
#      - An initial condition vector 'Y0' as a NumPy array
#      - A list of variable names 'var_names' matching len(Y0)
#      - An ODE function of the form: odes(t, y, params)
#
#  © 2025 Criseida G. Zamora Chimal
# =============================================================================

import numpy as np

# =========================
# Parameters
# =========================
params = {
    "k_prod": 5.0,   # production rate of X
    "k_deg": 0.5,    # degradation rate of X
    "alpha": 0.2,    # effect of X on Y production
    "beta": 0.1      # degradation rate of Y
}

# =========================
# Initial conditions
# =========================
# The system has two state variables:
#   X: regulatory species 
#   Y: downstream species 
Y0 = np.array([0.0, 0.0])  # [X0, Y0]

# =========================
# Variable names
# =========================
var_names = ["X", "Y"]

# =========================
# ODE system
# =========================
def odes(t, y, params):
    """
    Defines the right-hand side of the ODE system.

    Arguments:
        t : float
            Current time
        y : np.ndarray
            Current state vector [X, Y]
        params : dict
            Dictionary of model parameters

    Returns:
        dydt : np.ndarray
            Array of derivatives [dX/dt, dY/dt]
    """
    # Ensure non-negativity
    X, Y = np.maximum(y, 0.0)

    # Unpack parameters
    k_prod = params["k_prod"]
    k_deg  = params["k_deg"]
    alpha  = params["alpha"]
    beta   = params["beta"]

    # System of equations
    dX = k_prod - k_deg * X
    dY = alpha * X - beta * Y

    return np.array([dX, dY], dtype=float)
