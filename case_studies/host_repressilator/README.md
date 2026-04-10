# Host–Repressilator Hybrid Simulation

This case study reproduces the hybrid simulation of the 31-variable host–repressilator model from Weisse et al. (2015).

This directory is **self-contained** and can be executed independently of the main `src/` folder.

---

## ▶️ Run the Simulation

From this directory, run:

```bash
python hysimode.py --model host_repressilator.py --tfinal 2000 --runs 3
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
