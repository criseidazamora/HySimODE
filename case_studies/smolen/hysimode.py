# =============================================================================
#  hysimode.py
# -----------------------------------------------------------------------------
#  Description:
#      Core hybrid deterministic–stochastic simulator of the HySimODE framework.
#      Integrates ODE-based biochemical models while dynamically classifying
#      variables as deterministic or stochastic using a trained Random Forest
#      Classifier (RFC).
#
#  Functionality:
#      - Loads a model module (containing odes(), Y0, params, var_names)
#      - Runs an initial deterministic pre-simulation
#      - Extracts quantitative trajectory features from the ODE solution
#      - Applies the RFC model (via rfc_integration.py) to classify variables
#        into deterministic and stochastic regimes
#      - Executes hybrid simulation:
#            * Deterministic integration (solve_ivp)
#            * Stochastic simulation (SSA / Gillespie algorithm)
#      - Repeats for multiple stochastic realizations
#      - Saves all run outputs (CSV, TXT, plots) under 'output_hybrid/'
#
#  Inputs:
#      --model   : Path to a valid ODE model (e.g., models/smolen_odes.py)
#      --tfinal  : Total simulation time [default: 2000]
#      --dt      : Integration step (Δt) [default: 1.0]
#      --runs    : Number of stochastic realizations [default: 1]
#
#  Outputs:
#      output_hybrid/
#      ├── rfc_decisions.csv        # RFC-based classification per variable
#      ├── runs/results_run*.csv    # Trajectories per run
#      ├── timeseries_txt/          # Time + species data (plain text)
#      └── trajectory_*.png         # Plots per species
#
#  Notes:
#      - The RFC classifier and metadata must be located in the working directory
#        (rfc_calibrated.joblib and rfc_metadata.json).
#      - The model must expose:
#            odes(t, y, params), Y0 (array), params (dict), var_names (list)
#      - Solver options can be customized by defining 'solver_options' in the model.
#      - Consistent with RFC features defined in rfc_integration.py and
#        training pipeline (train_rfc.py).
#
#  Example:
#      # Case 1: Direct molecule-based model
#      python hysimode.py --model my_ode_model.py --tfinal 50 --dt 0.1 runs 3
#
#      # Case 2: Concentration-based model (via adapter)
#      BASE_MODEL=smolen_odes python hysimode.py --model concentration_adapter_hybrid.py --tfinal 500 --dt 0.5 --runs 1
#
#  © 2025 Criseida G. Zamora Chimal
# =============================================================================

import time
from time import perf_counter  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import importlib.util
import time
import random
import argparse
from rfc_integration import load_rfc, classify_species_with_rfc

# ==============================
# 1. Auxiliary functions
# ==============================

def load_model_from_file(model_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    odes = getattr(model, "odes")
    Y0 = getattr(model, "Y0")
    params = getattr(model, "params")
    var_names = getattr(model, "var_names")

    # Return also the module itself
    return odes, Y0, params, var_names, model


def reduced_system(odes, stochastic_indices, params):
    """Generate reduced ODE function (without explicit dynamics for stochastic species)."""
    def system(t, y):
        dy = odes(t, y, params)
        for idx in stochastic_indices:
            dy[idx] = 0.0
        return dy
    return system

def compute_propensities(odes, t, state, reaction_map, params,
                         model=None, atol_check=1e-6):

    """
    Compute SSA propensities from a deterministic ODE system.

    This function constructs the vector of reaction propensities required by
    a Stochastic Simulation Algorithm (SSA) using the drift of an ODE model.

    Two modes of operation are supported:

    1. If the provided model defines `odes_prod_deg(t, state, params)`,
    the function uses it to explicitly decompose the deterministic drift
    into production and degradation terms. Propensities are then obtained
    directly from these non-negative components.

    2. If `odes_prod_deg` is not available, the function falls back to using
    the standard ODE function `odes(t, state, params)` only. In this case,
    propensities are inferred from the sign of the drift:
    positive contributions correspond to production reactions and
    negative contributions correspond to degradation reactions.

    An optional consistency check can be performed to verify that

        odes(t, state, params) ≈ prod - deg

    when the decomposition function is available.

    Parameters
    ----------
    odes : callable
        ODE function of the model with signature `odes(t, y, params) -> dy`.
    t : float
        Current simulation time.
    state : np.ndarray
        Current state vector (concentrations or molecule counts).
    reaction_map : list[tuple[int, str]]
        List of pairs `(idx, 'prod'/'deg')` mapping each SSA reaction to
        a component of the ODE system.
    params : dict
        Dictionary containing the model parameters.
    model : module, optional
        Model module. If it defines `odes_prod_deg`, that function will be
        used to obtain explicit production and degradation terms.
    atol_check : float or None, optional
        Absolute tolerance for the consistency check between the ODE drift
        and the decomposition `prod - deg`. If None, the check is skipped.

    Returns
    -------
    np.ndarray
        One-dimensional array of propensities (all finite and ≥ 0).
    """

    state = np.asarray(state, dtype=float)

    if model is not None and hasattr(model, "odes_prod_deg"):
        try:
            # It tries to read: odes_prod_deg(t, y, params)
            prod, deg = model.odes_prod_deg(t, state, params)
        except TypeError:
            # Check if the user defined odes_prod_deg(t, y)
            prod, deg = model.odes_prod_deg(t, state)

        prod = np.asarray(prod, dtype=float)
        deg  = np.asarray(deg,  dtype=float)

        # --- Basic checks of  ---
        if prod.shape != deg.shape:
            raise ValueError(
                f"[ERROR] odes_prod_deg devolvió prod and deg with "
                f"different shapes: prod{prod.shape}, deg{deg.shape}"
            )

        if prod.ndim != 1:
            raise ValueError(
                f"[ERROR] odes_prod_deg debe devolver vectores 1D; "
                f"prod.ndim={prod.ndim}"
            )

        if prod.shape[0] != state.shape[0]:
            raise ValueError(
                f"[ERROR] Length of prod/deg ({prod.shape[0]}) differs "
                f"from the length of the state ({state.shape[0]})."
            )

        # --- Consistency checks: dy ≈ prod - deg ---
        if atol_check is not None:
            try:
                dy = odes(t, state, params)
            except TypeError:
                dy = odes(t, state)

            dy  = np.asarray(dy, dtype=float)
            net = prod - deg

            if dy.shape != net.shape:
                print(
                    "[WARN] Different shape between dy and (prod - deg): "
                    f"dy{dy.shape}, net{net.shape}"
                )
            else:
                if not np.allclose(dy, net, atol=atol_check, rtol=0.0):
                    max_diff = float(np.max(np.abs(dy - net)))
                    print(
                        "[WARN] Inconsistency between odes and odes_prod_deg: "
                        f"max |dy - (prod - deg)| = {max_diff:.3e} "
                        f"(atol={atol_check})"
                    )

        # --- Build propensities from prod/deg descomposition (Optional)---
        props = []
        n_vars = state.shape[0]

        for idx, rtype in reaction_map:
            if not (0 <= idx < n_vars):
                raise IndexError(
                    f"[ERROR] reaction_map contains and index out of range: "
                    f"idx={idx}, n_vars={n_vars}"
                )

            if rtype == 'prod':
                val = prod[idx]
            elif rtype == 'deg':
                val = deg[idx]
            else:
                raise ValueError(f"[ERROR] Unkwon type of reaction: {rtype}")

            
            val = float(val)
            if not np.isfinite(val):
                raise ValueError(
                    f"[ERROR] Infinite value on propensity ({rtype}) for "
                    f"idx={idx}: {val}"
                )

            props.append(max(val, 0.0))

    else:
        # ====================================
        # Calculates pripensities base on dy
        # ====================================
        try:
            dy = odes(t, state, params)
        except TypeError:
            # Models must contain odes as minimun requirement
            dy = odes(t, state)

        dy = np.asarray(dy, dtype=float)

        if dy.ndim != 1 or dy.shape[0] != state.shape[0]:
            raise ValueError(
                f"[ERROR] ODE returned an unexpected vector shape: "
                f"dy.shape={dy.shape}, len(state)={state.shape[0]}"
            )

        props = []
        n_vars = state.shape[0]

        for idx, rtype in reaction_map:
            if not (0 <= idx < n_vars):
                raise IndexError(
                    f"[ERROR] reaction_map contains an index out of range: "
                    f"idx={idx}, n_vars={n_vars}"
                )

            val = float(dy[idx])
            if not np.isfinite(val):
                raise ValueError(
                    f"[ERROR] Not finite value of dy for idx={idx}: {val}"
                )

            if rtype == 'prod':
                props.append(max(val, 0.0))
            elif rtype == 'deg':
                props.append(max(-val, 0.0))
            else:
                raise ValueError(f"[ERROR] Unkwon type of reaction: {rtype}")

    # ======================================================================
    # Sanity checks on vector propensities
    # ======================================================================
    props = np.asarray(props, dtype=float).ravel()

    if props.ndim != 1 or props.shape[0] != len(reaction_map):
        raise ValueError(
            "[ERROR] Unexpected shape of propensity vector: "
            f"{props.shape}, expected ({len(reaction_map)},)"
        )

    if not np.all(np.isfinite(props)):
        raise ValueError(
            "[ERROR] Propensities contains NaN or Inf. "
            f"props={props}"
        )

    if np.any(props < 0.0):
        # Check of non negativity
        raise ValueError(
            "[ERROR] Negative propensities after processing: "
            f"{props}"
        )

    return props

# =====================================
# 2. Principal motor of hybrid simulator
# =====================================

def run_hybrid_sim(model_path, T_FINAL=2000, DT=1.0, NUM_RUNS=1, OBS_WINDOW=100,
                   rfc_model="rfc_calibrated.joblib", rfc_meta="rfc_metadata.json"):
    """Run hybrid simulation with HySimODE."""
    total_start = perf_counter()
    # ---- Seeds----
    seed = int(time.time() * 1000) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)
    print(f"[INFO] Seed: {seed}")

    # ---- Load model ----
    odes, Y0, params, var_names, model = load_model_from_file(model_path)

    # Load solver options (model-specific or default)
    solver_options = getattr(model, "solver_options", {
        "method": "BDF",
        "rtol": 1e-6,
        "atol": 1e-9,
        "max_step": DT
    })

    # ==== Deterministic Pre-simulation  ====
    t_eval = np.linspace(0, T_FINAL, int(T_FINAL / DT) + 1)
    
    sol = solve_ivp(lambda t, y: odes(t, y, params),
                [0, T_FINAL], Y0,
                t_eval=t_eval,
                **solver_options)

    Y_det = np.maximum(sol.y, 0.0)

    # ==== Classification of RFC ====
    clf, feature_cols, best_th = load_rfc(rfc_model, rfc_meta)
    stochastic_indices, decisions_df = classify_species_with_rfc(
        sol.t, Y_det, clf, feature_cols, prob_threshold=best_th
    )

    print("[INFO] Stochastic species:", [var_names[i] for i in stochastic_indices])

    # Save RFC decisions
    os.makedirs("output_hybrid", exist_ok=True)
    decisions_df.to_csv("output_hybrid/rfc_decisions.csv", index=False)

    hybrid_start = perf_counter()
    # ==== Buid stequimetric matrix====
    stoich_matrix = []
    reaction_map = []
    for idx in stochastic_indices:
        v_prod = np.zeros(len(Y0)); v_prod[idx] = 1
        stoich_matrix.append(v_prod); reaction_map.append((idx, 'prod'))

        v_deg = np.zeros(len(Y0)); v_deg[idx] = -1
        stoich_matrix.append(v_deg); reaction_map.append((idx, 'deg'))

    stoich_matrix = np.array(stoich_matrix)

    # ==== Hybrid core ====
    averaged_results = {name: [] for name in var_names}
    last_run_T_vals, last_run_Y_vals = [], []

    for run in range(NUM_RUNS):
        print(f"[INFO] Run {run+1}/{NUM_RUNS}")
        
        run_start = perf_counter()
        t = 0.0
        state = np.array(Y0, dtype=float)
        T_vals, Y_vals = [t], [state.copy()]
        last_reported_minute = -1

        while t < T_FINAL:
            t_end = min(t + DT, T_FINAL)
            t_local = t

            # --- Gillespie SSA ---
            while True:
                
                #pass the 'model' object to use model.odes_prod_deg
                prop = compute_propensities(
                    odes,
                    t_local,
                    state,
                    reaction_map,
                    params,
                    model=model,     
                    atol_check=1e-6   # adjustable
                )

                a0 = np.sum(prop)
                if a0 <= 0:
                    break

                tau = np.random.exponential(1 / a0)
                if t_local + tau > t_end:
                    break

                r = np.random.rand() * a0
                reaction_index = np.searchsorted(np.cumsum(prop), r)
                state += stoich_matrix[reaction_index]
                state = np.maximum(state, 0)
                t_local += tau

            
            sol_block = solve_ivp(reduced_system(odes, stochastic_indices, params),
                      [t, t_end], state,
                      **solver_options)

            state = sol_block.y[:, -1]
            t = t_end

            # --- Save trajectory ---
            current_minute = int(t)
            if current_minute > last_reported_minute:
                last_reported_minute = current_minute

            T_vals.append(t)
            Y_vals.append(state.copy())

        # --- Mean of final window ---
        T_vals, Y_vals = np.array(T_vals), np.array(Y_vals)
        obs_mask = T_vals >= (T_FINAL - OBS_WINDOW)
        obs_Y = Y_vals[obs_mask]
        for key, idx in enumerate(var_names):
            averaged_results[idx].append(np.mean(obs_Y[:, key]))

            # === Save each run to CSV ===
        import pandas as pd
        out_dir = "output_hybrid/runs"
        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(Y_vals, columns=var_names)
        df.insert(0, "time", T_vals)
        run_file = os.path.join(out_dir, f"results_run{run+1}.csv")
        df.to_csv(run_file, index=False)
        print(f"[INFO] Saved run {run+1} to {run_file}")

        if run == NUM_RUNS - 1:
            last_run_T_vals, last_run_Y_vals = T_vals, Y_vals
        
        
        run_end = perf_counter()
        run_elapsed = run_end - run_start
        print(f"[TIME] Run {run+1} elapsed time: {run_elapsed:.3f} s")

    # ==== TIMING: end ====
    hybrid_end = perf_counter()
    total_end = perf_counter()

    hybrid_elapsed = hybrid_end - hybrid_start
    total_elapsed = total_end - total_start

    print(f"[TIME] Hybrid core total time: {hybrid_elapsed:.3f} s")
    print(f"[TIME] Full run_hybrid_sim time (incl. pre-sim + RFC): {total_elapsed:.3f} s")

    print("[INFO] Simulation finished. Results are saved in 'output_hybrid/'.")

    # ==== Save results====
    out_dir = "output_hybrid/timeseries_txt"
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(f"{out_dir}/time.txt", last_run_T_vals, fmt="%.3f")

    for i, name in enumerate(var_names):
        data = last_run_Y_vals[:, i]
        np.savetxt(f"{out_dir}/{name}.txt", data, fmt="%.3f")

    for i, name in enumerate(var_names):
        plt.figure()
        plt.plot(last_run_T_vals, last_run_Y_vals[:, i], label=name)
        plt.title(f"Trajectory of {name}")
        plt.xlabel("Time"); plt.ylabel(name)
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"output_hybrid/trajectory_{name}.png")
        plt.close()

    print("[INFO] Simulation finished. Results are saved in 'output_hybrid/'.")

# ==============================
# 3. CLI
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HySimODE hybrid simulation")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the file .py with ODEs, Y0, params")
    parser.add_argument("--tfinal", type=float, default=2000, help="Total time of simulation")
    parser.add_argument("--dt", type=float, default=1.0, help="Step time of simulation")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    args = parser.parse_args()

    run_hybrid_sim(args.model, T_FINAL=args.tfinal, DT=args.dt, NUM_RUNS=args.runs)
