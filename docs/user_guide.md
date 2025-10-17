# HySimODE User Guide

## 1. Introduction

**HySimODE** (Hybrid Simulator using ODEs and Random Forest Classification) is a framework for **hybrid deterministic–stochastic simulation** of biochemical systems.  
It automatically classifies each species as **deterministic or stochastic** using a **Random Forest Classifier (RFC)** trained from quantitative ODE features.

### Main components
| Module | Description |
|---------|--------------|
| `make_features_rfc.py` | Extracts quantitative time-series features from ODE trajectories |
| `train_rfc.py` | Trains the Random Forest Classifier (RFC) |
| `rfc_integration.py` | Loads and applies the RFC within hybrid simulations |
| `hysimode.py` | Core deterministic–stochastic hybrid simulator |
| `analyze_runs.py` | Aggregates and visualizes multiple simulation runs |

---

## 2. Requirements and Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/<yourname>/HySimODE-public.git
cd HySimODE-public
pip install -r requirements.txt
```

**Minimum requirements**
- Python ≥ 3.9  
- NumPy ≥ 1.24  
- SciPy ≥ 1.10  
- pandas ≥ 2.0  
- scikit-learn ≥ 1.3  
- matplotlib ≥ 3.8  

---

## 3. Model Preparation

Each biochemical model must be defined as a standalone Python module containing:

```python
Y0 = np.array([...])       # initial conditions
params = {...}             # model parameters
var_names = ["X", "Y", ...]  # variable names

def odes(t, y, params):
    # Example:
    dX = k_prod - k_deg * X
    dY = alpha * X - beta * Y
    return np.array([dX, dY])
```

The model **must export**:
- `Y0` (NumPy array)
- `params` (dictionary)
- `var_names` (list of species)
- `odes(t, y, params)` (function returning derivatives)

Optional:
- `solver_options` (dictionary for `solve_ivp`)
- compartment volumes (`vol_syn`, `vol_dend`, etc.)

A minimal example is provided in `models/my_ode_model.py`.

---

## 4. Running Hybrid Simulations

HySimODE can run **two types of models**:

### Case 1 — Molecule-based models
Run directly:

```bash
python hysimode.py --model models/my_ode_model.py --tfinal 100 --dt 0.1 --runs 3
```

### Case 2 — Concentration-based models (µM)
Use the adapter for conversion to molecule units:

```bash
BASE_MODEL=smolen_odes python hysimode.py     --model models/concentration_adapter_hybrid.py --tfinal 500 --dt 0.5 --runs 3
```

Here, `BASE_MODEL` refers to the source model defined in concentrations  
(e.g., `smolen_odes.py`), and the adapter handles the µM ↔ molecules conversion.

---

## 5. Output Files

Each simulation produces an organized output directory:

```
output_hybrid/
├── rfc_decisions.csv         # RFC-predicted stochastic vs deterministic species
├── runs/
│   ├── results_run1.csv
│   ├── results_run2.csv
│   └── ...
├── timeseries_txt/
│   ├── time.txt
│   ├── X.txt
│   ├── Y.txt
│   └── ...
└── trajectory_X.png, trajectory_Y.png, ...
```

**Descriptions**
- `rfc_decisions.csv`: probability and classification for each species  
- `runs/results_run*.csv`: full trajectories per stochastic realization  
- `timeseries_txt/*.txt`: time and species data for the final run  
- `trajectory_*.png`: plots of individual trajectories

---

## 6. Analyzing Results

Use `analyze_runs.py` to compute mean and standard deviation across runs:

```bash
python analyze_runs.py --dir output_hybrid/runs --plot
```

This generates:

```
summary/
├── species_summary.csv
└── plots/
    ├── X_avg_std.png
    ├── Y_avg_std.png
    └── ...
```

Each plot shows:
- Blue line → mean trajectory  
- Gray shaded band → ±1 standard deviation

---

## 7. RFC Model Integration

HySimODE automatically loads:
- `rfc_calibrated.joblib` → trained and calibrated Random Forest model  
- `rfc_metadata.json` → model metadata including:
  - feature list  
  - best hyperparameters  
  - optimal threshold for stochastic class

The features extracted during simulation are **identical** to those used in training:
`mean_total`, `std_total`, `cv_total`, `madydt_total`, `min`, `max`, `minmax_ratio`,  
`final_mean`, `final_std`, `final_cv`, `final_madydt`, `final_value`, `initial_mean`.

Detailed definitions are provided in `docs/feature_definitions.md`.

---

## 8. Tips and Troubleshooting

| Issue | Possible Cause | Solution |
|--------|----------------|-----------|
| Simulation diverges | Solver step too large | Reduce `dt` or set smaller `max_step` in `solver_options` |
| No stochastic species detected | Feature scaling mismatch | Ensure consistent feature list in JSON metadata |
| RFC not found | Missing `rfc_calibrated.joblib` | Copy RFC artifacts to the working directory |
| Adapter error | Missing environment variable | Set `BASE_MODEL=<your_model_name>` before running |

---

## 9. Citation

If you use HySimODE in scientific work, please cite:

> **Zamora Chimal, C.G. (2025).**  
> *HySimODE: A hybrid deterministic–stochastic ODE simulator with Random Forest classification.*  
> GitHub repository: [https://github.com/CriseidaZC/HySimODE-public](https://github.com/CriseidaZC/HySimODE-public)

---

## 10. License and Credits

HySimODE is released under the **MIT License**.

**Author:**  
© 2025 Criseida G. Zamora Chimal  
All rights reserved.

---
