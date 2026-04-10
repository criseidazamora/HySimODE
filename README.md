# HySimODE

HySimODE is a Python framework for hybrid stochastic–deterministic simulation of biochemical systems defined by ordinary differential equations (ODEs). It combines deterministic integration with stochastic updates and uses a pre-trained Random Forest classifier (RFC) to automatically assign species to stochastic or deterministic regimes.

This repository accompanies the publication:

**Criseida G. Zamora-Chimal and A.P.S. Darlington. HySimODE: A hybrid stochastic–deterministic simulation framework for multiscale models of biological systems. Bioinformatics.**

---

## Repository Structure

The repository is organized into four main components:

```text
HySimODE/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── hysimode.py
│   ├── analyze_runs.py
│   ├── concentration_adapter_hybrid.py
│   ├── rfc_integration.py
│   ├── rfc_calibrated.joblib
│   └── rfc_metadata.json
├── models/
│   ├── host_repressilator.py
│   ├── smolen_odes.py
│   ├── my_ode_model.py
│   ├── gene_expression.py
│   ├── gene_expression_odes_prod_deg.py
│   ├── ge_two_compartment.py
│   └── __init__.py
├── case_studies/
│   ├── host_repressilator/
│   │   ├── hysimode.py
│   │   ├── analyze_runs.py
│   │   ├── rfc_calibrated.joblib
│   │   ├── rfc_metadata.json
│   │   ├── rfc_integration.py
│   │   └── host_repressilator.py
│   └── smolen/
│       ├── hysimode.py
│       ├── analyze_runs.py
│       ├── rfc_calibrated.joblib
│       ├── rfc_metadata.json
│       ├── rfc_integration.py
│       ├── concentration_adapter_hybrid.py
│       └── smolen_odes.py
└── rfc_pipeline/
    ├── rfc_training/
    │   ├── models_training/
    │   │   ├── model1.py
    │   │   ├── ...
    │   │   └── model40.py
    │   ├── make_features_rfc.py
    │   ├── train_rfc.py
    │   ├── rfc_calibrated.joblib
    │   └── rfc_metadata.json
    ├── rfc_sensitivity/
    │   └── threshold_sensitivity_rfc.py
    ├── rfc_testing_unseen_models/
    │   ├── host_repressilator/
    │   │   ├── make_features_host_repressilator.py
    │   │   └── predict_with_rfc_on_host_repressilator.py
    │   └── smolen/
    │       ├── make_features_smolen.py
    │       └── predict_with_rfc_on_smolen.py
    └── model_dataset_audit/
        ├── audit_family_similarity.py
        └── audit_models.py
```

---

### 1. `src/` — Core framework

Contains the main HySimODE implementation, including the simulation engine, classifier integration, and supporting utilities.

* `hysimode.py` — hybrid simulation engine
* `analyze_runs.py` — post-processing tools
* `concentration_adapter_hybrid.py` — adapter for concentration-based models
* `rfc_integration.py` — classifier integration
* `rfc_calibrated.joblib`, `rfc_metadata.json` — trained classifier

Files located outside this directory will not be detected by the pipeline.

---

### 2. `models/` — Reference and example model definitions

```text
models/
├── host_repressilator.py
├── smolen_odes.py
├── my_ode_model.py
├── gene_expression.py
├── gene_expression_odes_prod_deg.py
├── ge_two_compartment.py
└── __init__.py
```

Provides a collection of model definitions used across the repository, including:

* `host_repressilator.py` — model used in the host–repressilator case study
* `smolen_odes.py` — model used in the Smolen case study
* `my_ode_model.py` — minimal structure that any ODE-based system must follow
* `gene_expression.py` — deterministic ODE benchmark used by HySimODE in net-drift mode (no explicit production/degradation terms)
* `gene_expression_odes_prod_deg.py` — HySimODE benchmark with explicit production/degradation terms via `odes_prod_deg(t, y, params)`
* `ge_two_compartment.py` — validation example of the concentration-to-molecule adapter in a multi-compartment system (demonstrates automatic, model-agnostic unit conversion)

These files serve as reusable references and templates for user-defined models.

---

### 3. `case_studies/` — Reproducibility of published results

Each case study is provided as a **self-contained execution environment**, including:

* the HySimODE simulation engine
* classifier files
* model definitions

These folders reproduce the results and figures presented in the paper.

Available case studies:

* `host_repressilator/`
* `smolen/`

Each folder includes a local `README.md` with instructions.

---

### 4. `rfc_pipeline/` — Classifier training and evaluation

Contains the full pipeline used to train, validate, and analyze the Random Forest classifier (RFC), including:

* feature generation and model training (`rfc_training/`)
* sensitivity analysis (`rfc_sensitivity/`)
* evaluation on unseen models (`rfc_testing_unseen_models/`)
* dataset auditing and analysis (`model_dataset_audit/`)

This component ensures full methodological reproducibility of the classification step.

---

**Note:** The `models/` directory provides reusable model definitions, while `case_studies/` contains self-contained execution environments used to reproduce the published results.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/criseidazamora/HySimODE.git
cd HySimODE
pip install -r requirements.txt
```

---

## Running HySimODE (general usage)

This corresponds to applying a pre-trained RFC model included in the repository.

To run HySimODE on a custom ODE model:

1. Place your model file in the same directory as `hysimode.py` (i.e., inside `src/`).

2. Run:

```bash
cd src/
python hysimode.py --model my_ode_model.py --tfinal 1000 --dt 1.0 --runs 3
```

**Important:** The HySimODE pipeline requires the model file and classifier files to be located in the same directory as the simulation engine. Files located outside this directory will not be detected by the pipeline.

---

## 🧬 Extending to New Models

You can adapt HySimODE to your own biochemical model by adding a new Python file (ODE model) in the same directory as `hysimode.py` (e.g., inside `src/`).

Each model file must define the following elements:

* `odes(t, y, params)` — defines the ODE system
* `params` — dictionary of model parameters
* `Y0` — vector of initial conditions
* `var_names` — list of variable names corresponding to the state vector

All output (trajectories, ensemble statistics, RFC decisions, and plots) will be automatically generated inside the `output_hybrid/` directory.

The template ODE model provided in `models/my_ode_model.py` defines the minimal structure that any ODE-based system must follow to be compatible with HySimODE.

All models must follow the interface described in the Supplementary Information (Section 6).

---

## Example: Host–Repressilator case study

```bash
cd case_studies/host_repressilator/
python hysimode.py --model host_repressilator.py --tfinal 1000 --dt 1.0 --runs 3
```

---

## Example: Smolen model (concentration-based)

```bash
cd case_studies/smolen/
BASE_MODEL=smolen_odes python hysimode.py --model concentration_adapter_hybrid.py --tfinal 460 --dt 0.01 --runs 3
```

---

## Output

Simulation outputs are saved in structured directories (e.g., `output_hybrid/`) within each run context. Post-processing can be performed using `analyze_runs.py`.

---

## Documentation

A complete user guide, including model structure, workflow, and execution details, is provided in:

**Supplementary Information (Section 6)** of the associated publication.

For platform-specific instructions (e.g., Windows), refer to the Supplementary Information.

---

## DOI

A permanent DOI for this software release is provided via Zenodo:

**DOI: (to be added after release)**

---

## License

This project is released under the MIT License.

---

## Contact

For questions or support, please contact:

[criseida.zamora@warwick.ac.uk](mailto:criseida.zamora@warwick.ac.uk)
