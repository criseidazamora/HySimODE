# 🧠 RFC Training and Testing Pipeline for HySimODE

This directory contains all scripts, data structures, and procedures used to train, calibrate, and testing the **Random Forest Classifier (RFC)** employed in **HySimODE** to assign biochemical species to *deterministic* or *stochastic* simulation regimes.

## Contents
- `train_rfc.py` – Training and calibration of the classifier.
- `make_features_rfc.py` – Feature extraction from training models.
- `rfc_metadata.json` – Stored metadata (best parameters, threshold, metrics, features).
- `rfc_calibrated.joblib` – Calibrated classifier used in all subsequent analyses.

---

## 🧠 Purpose & Reproducibility

This pipeline ensures **full reproducibility** of the machine learning component described in the HySimODE manuscript.  
The trained RFC learns from quantitative features extracted from multiple ODE-based biochemical models and generalizes to unseen systems.

- **Goal:** Automate species regime classification (deterministic vs. stochastic) using abundance-derived features.
- **Approach:** Random Forest Classifier (RFC) trained on nine mechanistic ODE models.
- **Validation:** Independent testing on two unseen systems — the *host–repressilator* model and the *Smolen* synaptic tagging model.
- **Reproducibility:** All scripts, configurations, and intermediate outputs are versioned and open.

---

## ⚙️ Requirements

Install the following dependencies before running any script:

```bash
pip install numpy pandas scikit-learn joblib matplotlib
```

Additional built-in modules used:
`os`, `json`, `typing`, `importlib.util`

---

## 🧩 Training the RFC (with 9 ODE models)

All training scripts are contained in the folder:

```
rfc_pipeline/rfc_training/
```

### Structure

```
rfc_training/
├── models_training/                # 9 ODE models used for supervised training
│   ├── model1.py
│   ├── ...
│   └── model9.py
│
├── make_features_training.py       # Simulates ODE models and extracts statistical features
├── train_rfc.py                    # Trains and calibrates the RFC (produces .joblib & .json)
├── rfc_calibrated.joblib           # Final trained classifier
├── rfc_metadata.json               # Feature metadata (names, calibration, parameters)
└── feature_definitions.md          # Description of extracted features
```

### Training Workflow

1. **Generate features**
   ```bash
   python generate_features_training.py
   ```
   This script simulates the nine training ODE systems and outputs a combined CSV (`features_rfc.csv`) with quantitative descriptors and labels.

2. **Train the Random Forest**
   ```bash
   python train_rfc.py
   ```
   The script performs:
   - A 75/25 train–test split (stratified)
   - 5-fold cross-validation with randomized search for hyperparameters
   - Probability calibration (isotonic regression)
   - Prints validation metrics (Accuracy, ROC-AUC, F1) to console
   - Saves the final model as:
     - `rfc_calibrated.joblib`
     - `rfc_metadata.json`

---

## 🧪 Testing on Unseen Models (host_repressilator, smolen)

Independent generalization testing is located in:

```
rfc_pipeline/rfc_testing_unseen_models/
```

### Structure

```
rfc_testing_unseen_models/
├── host_repressilator/
│   ├── make_features_host_repressilator.py     # Generates feature CSV for the repressilator model
│   └── predict_with_rfc_on_host_repressilator.py  # Applies the trained RFC → rfc_decisions.csv
│
├── smolen/
│   ├── make_features_smolen.py
│   └── predict_with_rfc_on_smolen.py
│
├── results/                                   # Stores all prediction CSVs
└── README.md                                  # Local usage instructions
```

### Example Workflow

#### Host–Repressilator
```bash
python make_features_host_repressilator.py
python predict_with_rfc_on_host_repressilator.py
```

Generates:
- `features_host_repressilator.csv`
- `predictions_host_repressilator.csv` → contains predicted class labels and probabilities

#### Smolen Model
```bash
python make_features_smolen.py
python predict_with_rfc_on_smolen.py
```

Generates:
- `features_smolen.csv`
- `predictions_smolen.csv`

---

## 📊 Output Structure and Interpretation

### RFC Training Outputs
- `rfc_calibrated.joblib` — trained classifier  
- `rfc_metadata.json` — feature info, calibration parameters  
- `features_rfc.csv` — full training dataset  
- Console output: cross-validation metrics and feature ranking (Gini importance)

### RFC Testing Outputs
- `features_<model>.csv` — computed features for each unseen model  
- `predictions_<model>.csv` — per-species regime label (0 = deterministic, 1 = stochastic) and probability score  
- Optional plots of feature importance can be generated with the provided utility scripts.

---

This pipeline was developed as part of the **HySimODE** framework within the *Control and Biological Engineering Group, University of Warwick*.

**Author:** Criseida Zamora  
**Contact:** For questions or comments, please open a new issue.  
(c) 2025 HySimODE Project. 

If you use this machine learning pipeline or its results, please cite this repository.

---
