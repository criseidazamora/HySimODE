# Feature Definitions (RFC Pipeline)

This document defines the **13 quantitative trajectory features** used to train and apply the Random Forest Classifier (RFC) in HySimODE. The **names, ordering, and computation** here match the artifacts in `rfc_metadata.json` and the extraction scripts (`make_features_rfc.py`, `make_features_host_repressilator.py`, `make_features_smolen.py`).

**Feature order (must match exactly):**

1. `mean_total`  
2. `std_total`  
3. `cv_total`  
4. `madydt_total`  
5. `min`  
6. `max`  
7. `minmax_ratio`  
8. `final_mean`  
9. `final_std`  
10. `final_cv`  
11. `final_madydt`  
12. `final_value`  
13. `initial_mean`

> The RFC expects inputs in **this exact order** (see `rfc_metadata.json["features"]`). Any change requires retraining.

---

## Global conventions

- Let \( t \in \mathbb{R}^{N} \) be the time grid (monotone increasing), \( Y \in \mathbb{R}^{M \times N} \) the simulated trajectories (rows = species/variables).
- **Non-negativity:** features are computed after clipping trajectories element-wise: `Y = np.maximum(Y, 0.0)`.
- **Windows:**  
  - Observation (final) window: last **100.0** time units: \( t \ge t_{\text{end}} - 100.0 \).  
  - Initial window: first **50.0** time units: \( t \le 50.0 \).
- **Derivative:** \(\frac{dy}{dt}\) estimated with `np.gradient(y, t)`.
- **Numerical guard:** divisions use \(\varepsilon = 1\times10^{-12}\) to avoid zero-division.
- **Units:** features are computed on the same scale as the input `Y`. If an adapter converts concentrations→molecules (e.g., Smolen), features reflect that scale. Ratios (e.g., `cv_*`, `minmax_ratio`) are scale-invariant; means/stds are not.

---

## Feature definitions

Let \( y \) be a single species trajectory, \( dy/dt \) its gradient, and define:
- Final window mask \( \mathcal{F} = \{ t \mid t \ge t_{\text{end}} - 100 \} \).
- Initial window mask \( \mathcal{I} = \{ t \mid t \le 50 \} \).

| Name | Definition | Notes |
|---|---|---|
| `mean_total` | \( \mathrm{mean}(y) \) over full trajectory | Sensitive to scale. |
| `std_total` | \( \mathrm{std}(y) \) over full trajectory | Sensitive to scale. |
| `cv_total` | \( \frac{\mathrm{std}(y)}{\mathrm{mean}(y)+\varepsilon} \) | Dimensionless. |
| `madydt_total` | \( \mathrm{mean}(|\frac{dy}{dt}|) \) over full trajectory | Uses `np.gradient(y, t)`. |
| `min` | \( \min(y) \) over full trajectory | After non-negativity clipping. |
| `max` | \( \max(y) \) over full trajectory | After non-negativity clipping. |
| `minmax_ratio` | \( \frac{\min(y)+\varepsilon}{\max(y)+\varepsilon} \) | \(\in(0,1]\) if `max>0`. |
| `final_mean` | \( \mathrm{mean}(y_{\mathcal{F}}) \) | Final 100 time units. |
| `final_std` | \( \mathrm{std}(y_{\mathcal{F}}) \) |  |
| `final_cv` | \( \frac{\mathrm{std}(y_{\mathcal{F}})}{\mathrm{mean}(y_{\mathcal{F}})+\varepsilon} \) |  |
| `final_madydt` | \( \mathrm{mean}(|\frac{dy}{dt}|_{\mathcal{F}}) \) |  |
| `final_value` | Last sample: \( y[-1] \) | Scalar. |
| `initial_mean` | \( \mathrm{mean}(y_{\mathcal{I}}) \) | First 50 time units. |

---

## Labeling rule (used in training)

For training labels (`label ∈ {0,1}` with 1 = stochastic):

\[
\texttt{label} = \mathbf{1}\Big( \texttt{final\_mean} < 200 \;\land\; \texttt{mean\_total} < 200 \;\land\; \texttt{initial\_mean} < 200 \Big)
\]

This rule is implemented in `make_features_rfc.py` and is **part of the dataset generation**, not of the RFC prediction.

---

## Reference implementation (minimal, consistent)

```python
import numpy as np

EPS = 1e-12
OBS_WINDOW = 100.0
INITIAL_WINDOW = 50.0

def compute_13_features(t, y):
    y = np.maximum(np.asarray(y, float), 0.0)
    t = np.asarray(t, float)
    dy = np.gradient(y, t)

    tend = t[-1]
    fmask = t >= (tend - OBS_WINDOW)
    imask = t <= INITIAL_WINDOW

    # whole
    mean_total = float(np.mean(y))
    std_total  = float(np.std(y))
    cv_total   = float(std_total / (mean_total + EPS))
    madydt_total = float(np.mean(np.abs(dy)))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    minmax_ratio = float((ymin + EPS) / (ymax + EPS))

    # final window
    yf, dyf = y[fmask], dy[fmask]
    final_mean    = float(np.mean(yf))
    final_std     = float(np.std(yf))
    final_cv      = float(final_std / (final_mean + EPS))
    final_madydt  = float(np.mean(np.abs(dyf)))
    final_value   = float(y[-1])

    # initial window
    yi = y[imask]
    initial_mean = float(np.mean(yi))

    return [
        mean_total, std_total, cv_total, madydt_total,
        ymin, ymax, minmax_ratio,
        final_mean, final_std, final_cv, final_madydt,
        final_value, initial_mean
    ]
```

> In multi-species settings, apply this function **row-wise** on `Y[i, :]` for each species.

---

## Consistency requirements (training, testing, integration)

- Use **exactly these 13 features** and **this order** everywhere.  
  The authoritative list is in `rfc_metadata.json["features"]`.
- Apply the **same windows/gradient/epsilon** and **non-negativity clipping** before feature computation.
- In prediction:
  - Load `rfc_calibrated.joblib`.
  - Compute calibrated probabilities and apply the **exact JSON threshold** `best_threshold_for_class1` (e.g., 0.05).
  - Build `X` strictly as `df[meta["features"]]`; do **not** add/remove/reorder columns.

---

## Units and adapters

- If a model is defined in **concentrations** (µM) but integrated in **molecules** via an adapter (e.g., Smolen with `concentration_adapter_hybrid.py`), features reflect that scale.  
- Ratio features (`cv_*`, `minmax_ratio`) are scale-invariant; mean/std/value based features are not. Keep the **same adapter** choice used during testing to avoid distribution drift.

---

## Validation checklist

- [ ] CSV contains exactly these 13 columns (in any order) plus optional non-numeric metadata.  
- [ ] When constructing `X`, select columns by **name and order** from `rfc_metadata.json["features"]`.  
- [ ] No extra numeric columns leak into `X`.  
- [ ] Final prediction uses `predict_proba(... )[:,1] >= best_threshold_for_class1`.  
- [ ] For Smolen/host–repressilator, the adapter and solver settings are consistent across feature generation and prediction.

---

## Change policy

Any change to:
- feature **definitions**,
- **order**,
- windows (`OBS_WINDOW`, `INITIAL_WINDOW`),
- preprocessing, or
- unit conventions

**invalidates the trained classifier**. You must regenerate the dataset, retrain the RFC, and update `rfc_metadata.json` accordingly. Since your artifacts are **frozen for publication**, do **not** modify this file unless you intend to retrain.

---

*Version:* 1.0 (matches `rfc_metadata.json` used for the published results)
