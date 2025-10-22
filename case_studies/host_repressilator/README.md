# Host–Repressilator Hybrid Simulation Example

This example reproduces the hybrid stochastic–deterministic simulation of the 31-variable host–repressilator system described in Weisse et al. (2015).

---

## 🧠 Description
The model couples a coarse-grained host physiology module with a three-gene repressilator circuit.  
Hybrid dynamics are simulated by combining deterministic ODE integration for high-copy species and stochastic updates for low-copy species as classified by the Random Forest Classifier (RFC).

---

## ▶️ Run the Simulation

From this directory, simply run:

```bash
python hysimode.py --model host_repressilator --tmax 2000 --runs 3
```

This will:
- Load the model from `host_repressilator.py`
- Use the pre-trained classifier `rfc_calibrated.joblib`
- Perform 3 independent hybrid runs (with random seeds)
- Store all results under `output_hybrid/`

---

## 📊 Post-processing and Visualization

To compute ensemble statistics and generate variability plots:

```bash
python analyze_runs.py --dir output_hybrid/runs --plot
```

This will create:
- `species_summary.csv` (mean and std for each species)
- Plots under `output_hybrid/summary/plots/`

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

---

## ⚙️ Requirements

Ensure the following Python packages are installed:

```bash
pip install -r ../../requirements.txt
```

---

## 📘 Citation

If you use this example or the HySimODE framework, please cite:

> Zamora Chimal et al, *HySimODE: A Hybrid Deterministic–Stochastic Framework for Multiscale Biochemical Simulation*, 2025.
