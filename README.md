# HySimODE: Hybrid Stochastic–Deterministic Simulation Framework

**HySimODE** is an open-source computational framework for **hybrid stochastic–deterministic simulation** of biochemical systems.  
It integrates a **Random Forest Classifier (RFC)** that automatically determines which species should be simulated stochastically or deterministically based on abundance-derived features.

---

## 🚀 Key Features

- **Automatic stochastic/deterministic partition** via RFC classification  
- **Hybrid time integration** coupling stochastic updates with ODE solvers  
- **Plug-and-play ODE models** — easily extend with new systems  
- **Post-processing utilities** for ensemble statistics and visualization  
- **Validated on two multiscale benchmarks**:  
  - Host–Repressilator system (Weisse *et al.*, 2015)  
  - Smolen synaptic tagging model (Smolen *et al.*, 2012)

---

## 🧰 Installation

Clone this repository and install required dependencies:

```bash
git clone https://github.com/criseidazamora/HySimODE-public.git
cd HySimODE-public
pip install -r requirements.txt
```

---

## 🧪 Quick Start

Run a hybrid simulation directly from the command line (CLI):

```bash
python src/hysimode.py --model models/my_ode_model.py --tmax 500 --runs 3
```

All outputs (CSV files, plots, and ensemble statistics) will be generated in:

```
output_hybrid/
│
├── runs/
├── summary/
├── rfc_decisions.csv
├── timeseries_txt/
└── species_plots/
```

To analyze ensemble variability and generate mean ± SD plots:

```bash
python src/analyze_runs.py
```

---

## 📂 Repository Structure

```
HySimODE-public/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   ├── hysimode.py
│   ├── rfc_integration.py
│   ├── concentration_adapter_hybrid.py
│   ├── analyze_runs.py
│   └── __init__.py
│
├── models/
│   ├── host_repressilator.py
│   ├── smolen_odes.py
│   ├── my_ode_model.py   # minimal structure that any ODE-based system must follow 
│   └── __init__.py
│
├── rfc_pipeline/
│   ├── rfc_training/
│   │   ├── models_training/
│   │   │   ├── model1.py
│   │   │   ├── ...
│   │   │   └── model9.py
│   │   ├── generate_features_training.py
│   │   ├── train_rfc.py
│   │   ├── rfc_calibrated.joblib
│   │   ├── rfc_metadata.json
│   │   └── feature_definitions.md
│   │
│   ├── rfc_testing_unseen_models/
│   │   ├── host_repressilator/
│   │   │   ├── make_features_host_repressilator.py
│   │   │   └── predict_with_rfc_on_host_repressilator.py
│   │   ├── smolen/
│   │   │   ├── make_features_smolen.py
│   │   │   └── predict_with_rfc_on_smolen.py
│   │   ├── results/
│   │   
│   └── README.md
│
├── case_studies/
│   ├── host_repressilator/
│   │   ├── hysimode.py
│   │   ├── rfc_integration.py
│   │   ├── concentration_adapter_hybrid.py
│   │   ├── analyze_runs.py
│   │   ├── host_repressilator.py
│   │   ├── rfc_calibrated.joblib
│   │   ├── rfc_metadata.json
│   │   ├── output_hybrid/
│   │   └── README.md
│   │
│   ├── smolen/
│   │   ├── hysimode.py
│   │   ├── rfc_integration.py
│   │   ├── concentration_adapter_hybrid.py
│   │   ├── analyze_runs.py
│   │   ├── smolen_odes.py
│   │   ├── rfc_calibrated.joblib
│   │   ├── rfc_metadata.json
│   │   ├── output_hybrid/
│   │   └── README.md
│   │
│   └── quickstart_cli.md
│
└── docs/
    ├── user_guide.md
    └── supplementary_information.pdf

```

---

## 🧬 Extending to New Models

You can easily adapt **HySimODE** to your own biochemical model by adding a new Python file (ODE model) in the **same directory as `hysimode.py`**.  
Each model file must define the following elements:

- `odes(t, y, params)` — defines the ODE system  
- `params` — dictionary of model parameters  
- `Y0` — vector of initial conditions  
- `var_names` — list of variable names corresponding to the state vector  

Example usage (run from the `src/` folder):

```bash
python hysimode.py --model my_ode_model.py --tmax 1000 --runs 3

All output (trajectories, ensemble statistics, RFC decisions, and plots) will be automatically generated inside the output_hybrid/ directory.

Note:
The current version expects the model file to be located in the same folder as hysimode.py.

The template ODE model (models/my_ode_model.py) defines the minimal structure that any ODE-based system must follow to be compatible with HySimODE.

## 📘 Documentation
 
- **Supplementary Information:** [`docs/supplementary_information.pdf`](docs/supplementary_information.pdf)  
- **Feature definitions:** [`rfc_training_testing/feature_definitions.md`](rfc_training_testing/feature_definitions.md)  
- **Case studies:** [`case_studies/`](case_studies/)

---

HySimODE is a research project from the Control and Biological Engineering Group (University of Warwick).

Lead developer: Criseida G. Zamora
Contact: For comments or inquiries, please open a new issue.
© 2025 HySimODE Project.

---
## 📜 License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, suggestions, and pull requests are welcome!  
Please follow standard GitHub contribution practices or open an issue for discussion.

---

## 🧩 Contact

For questions or collaborations, please open an issue on [GitHub Issues](https://github.com/cg-zamora/HySimODE-public/issues).
