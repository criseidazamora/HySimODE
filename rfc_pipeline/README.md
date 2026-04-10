# RFC Pipeline for HySimODE

This directory contains all scripts, data structures, and procedures used to train, calibrate, and evaluate the **Random Forest classifier (RFC)** employed in **HySimODE** to assign biochemical species to *deterministic* or *stochastic* simulation regimes.

---

## Directory Structure

The `rfc_pipeline/` directory is organized into four main components:

```text
rfc_pipeline/
├── rfc_training/
│   ├── models_training/
│   │   ├── model1.py
│   │   ├── ...
│   │   └── model40.py
│   ├── make_features_training.py
│   ├── train_rfc.py
│   ├── rfc_calibrated.joblib
│   ├── rfc_metadata.json
│   └── feature_definitions.md
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

## Purpose and Reproducibility

This pipeline ensures **full reproducibility** of the machine learning component described in the HySimODE manuscript.

The trained RFC learns from quantitative features extracted from deterministic simulations of biochemical ODE models and is then applied to unseen systems.

* **Goal:** automate species regime classification (deterministic vs. stochastic)
* **Approach:** Random Forest classifier trained on **40 mechanistic ODE models**
* **Validation:** tested on two unseen systems — the *host–repressilator* model and the *Smolen* synaptic tagging model
* **Reproducibility:** all scripts, configurations, and outputs required to reproduce training and evaluation are included

---

## Requirements

Install the required dependencies before running any script:

```bash
pip install numpy pandas scipy scikit-learn joblib matplotlib tabulate
```

Additional built-in modules used by the scripts include:

* `os`
* `json`
* `typing`
* `importlib.util`

---

## Important Execution Note

The training and feature-generation scripts expect the relevant files to be located in the same working directory where the corresponding script is executed.

In particular:

* the training models in `models_training/` must remain available in the same location of `train_rfc.py` to run the evaluation metrics of the RFC.
* scripts in the testing folders should be run from their own local directories

---

## 1. `rfc_training/` — Training and calibration of the classifier

This folder contains the full supervised training workflow for the RFC.

### Contents

* `models_training/` — collection of **40 ODE models** used for RFC training
* `make_features_training.py` — simulates the training models and extracts quantitative features
* `train_rfc.py` — trains and calibrates the Random Forest classifier
* `rfc_calibrated.joblib` — final trained classifier
* `rfc_metadata.json` — stored metadata (features, calibration, parameters, metrics)

---

### Workflow

#### Step 1 — Generate features

Run from `rfc_training/`:

```bash
python make_features_training.py
```

This script simulates the 40 training ODE systems and generates a combined feature table (e.g., `features_rfc.csv`) containing quantitative descriptors and labels.

---

#### Step 2 — Train the classifier

```bash
python train_rfc.py
```

This script trains a Random Forest classifier to predict whether each biochemical species should be simulated stochastically (1) or deterministically (0), using features derived from deterministic simulations.

The training pipeline includes:

* hyperparameter optimization
* probability calibration
* evaluation under cross-validation
* leave-one-model-out (LOMO) analysis, when enabled

Optional robustness analyses may include:

* nested LOMO hyperparameter search
* bootstrap confidence intervals over held-out models
* permutation tests

---

### Main outputs

* `rfc_calibrated.joblib`
* `rfc_metadata.json`
* `features_rfc.csv`
* optional evaluation summaries such as:

  * `lomo_metrics_by_model.csv`
  * `lomo_summary.json`

---

## 2. `rfc_sensitivity/` — Threshold sensitivity analysis

This folder contains scripts used to assess the robustness of the classifier with respect to the abundance threshold used to define stochastic vs. deterministic labels.

### Contents

* `threshold_sensitivity_rfc.py` — evaluates classifier performance under alternative abundance thresholds

---

### Workflow

Run from `rfc_sensitivity/`:

```bash
python threshold_sensitivity_rfc.py
```

This script re-evaluates the classifier under different threshold settings and summarizes how label assignment and model performance change across thresholds.

---

### Main outputs

Typical outputs may include:

* threshold-wise performance summaries
* comparison tables for label changes
* robustness metrics across thresholds

---

## 3. `rfc_testing_unseen_models/` — Testing on unseen systems

This folder contains scripts used to evaluate RFC generalization on biochemical systems not used during training.

---

### Host–Repressilator

Run from `rfc_testing_unseen_models/host_repressilator/`:

```bash
python make_features_host_repressilator.py
python predict_with_rfc_on_host_repressilator.py
```

---

### Smolen model

Run from `rfc_testing_unseen_models/smolen/`:

```bash
python make_features_smolen.py
python predict_with_rfc_on_smolen.py
```

---

### Outputs

* `features_<model>.csv`
* `predictions_<model>.csv`

These files contain per-species probabilities and assigned regime labels.

---

## 4. `model_dataset_audit/` — Dataset audit and model similarity analysis

This folder contains scripts used to inspect and validate the training dataset.

### Contents

* `audit_models.py` — audits model structure and extracted features
* `audit_family_similarity.py` — evaluates similarity relationships across model families

---

### Workflow

Run from `model_dataset_audit/`:

```bash
python audit_models.py
python audit_family_similarity.py
```

These scripts support dataset validation, diversity assessment, and model curation.

---

## Output Summary

Across the full `rfc_pipeline/`, the main outputs include:

* `features_rfc.csv` — training feature table
* `rfc_calibrated.joblib` — trained classifier
* `rfc_metadata.json` — metadata and configuration
* `features_<model>.csv` — extracted features for unseen models
* `predictions_<model>.csv` — probabilities and predicted labels
* optional sensitivity and audit summaries

---

## Citation

If you use this machine learning pipeline or its results, please cite the associated HySimODE publication.
