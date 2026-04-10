# Smolen Synaptic Tagging Hybrid Simulation

This case study reproduces the hybrid simulation of the 23-variable synaptic tagging and long-term potentiation (LTP) model from Smolen et al. (2012).

This directory is **self-contained** and can be executed independently of the main `src/` folder.

---

## ▶️ Run the Simulation

From this directory, run:

```bash
BASE_MODEL=smolen_odes python hysimode.py --model concentration_adapter_hybrid.py --tfinal 500 --dt 0.5 --runs 3
```

This will generate multiple hybrid trajectories using the pre-trained classifier included in this folder.

---

## 📊 Post-processing

To compute ensemble statistics and generate plots:

```bash
python analyze_runs.py --dir output_hybrid/runs --plot
```

---

## 📁 Output

Results are stored in:

```text
output_hybrid/
├── runs/
├── summary/
├── rfc_decisions.csv
└── plots/
```

---

## ⚙️ Requirements

Install dependencies from the root directory:

```bash
pip install -r ../../requirements.txt
```

---

## 📘 Citation

If you use this case study, please cite the associated HySimODE publication.
