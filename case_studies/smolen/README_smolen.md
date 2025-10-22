# Smolen Synaptic Tagging Hybrid Simulation Example

This example reproduces the hybrid stochastic–deterministic simulation of the 23-variable synaptic tagging and long-term potentiation (LTP) model described in Smolen et al. (2012).

---

## 🧠 Description

The Smolen model represents a biochemical signaling cascade underlying synaptic tagging and capture in long-term memory formation.  
It includes multiple kinase activation loops (e.g., CaMKII, PKA, ERK) and the synthesis of transient tags that mark synapses for protein capture.

Hybrid simulations in **HySimODE** combine deterministic integration for abundant molecular species with stochastic updates for low-copy intermediates, such as `tagd1` and `tagd2`, as classified by the Random Forest Classifier (RFC).

---

## ▶️ Run the Simulation

From this directory, run:

```bash
BASE_MODEL=smolen_odes python hysimode.py --model concentration_adapter_hybrid.py --tfinal 500 --dt 0.5 --runs 3
```

This will:
- Load the model from `smolen_odes.py`
- Use the pre-trained RFC classifier (`rfc_calibrated.joblib`)
- Execute 3 independent hybrid realizations (with different random seeds)
- Store all results in the folder `output_hybrid/`

---

## 📊 Post-processing and Visualization

To compute ensemble statistics and plot the variability across hybrid runs:

```bash
python analyze_runs.py --dir output_hybrid/runs --plot
```

This script will generate:
- `species_summary.csv` — mean and standard deviation for each species  
- Variability plots under `output_hybrid/summary/plots/`

---

## 📁 Output Structure

```
output_hybrid/
├── runs/                # results_run1.csv, results_run2.csv, ...
├── summary/
│   ├── species_summary.csv
│   └── plots/
├── rfc_decisions.csv
├── timeseries_txt/
└── plots/
```

Each hybrid run includes all 23 variables.  
Stochastic components such as `tagd2` exhibit amplitude and timing variability, while deterministic components remain smooth and consistent across runs.

---

## ⚙️ Requirements

Ensure the following dependencies are installed before running:

```bash
pip install -r ../../requirements.txt
```

---

## 📘 Citation

If you use this example or the HySimODE framework, please cite:

> Zamora Chimal et al., *HySimODE: A Hybrid Deterministic–Stochastic Framework for Multiscale Biochemical Simulation*, 2025.

---
