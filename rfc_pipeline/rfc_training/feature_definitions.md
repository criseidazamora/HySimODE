# Feature Definitions (RFC Pipeline)

This document defines the **13 quantitative trajectory features** used to train and apply the Random Forest Classifier (RFC) in HySimODE. The **names, ordering, and computation** here match the artifacts in `rfc_metadata.json` and the extraction scripts (`make_features_rfc.py`, `make_features_host_repressilator.py`, `make_features_smolen.py`).

**Feature order (must match exactly):**

1. `initial_q50`  
2. `initial_cv`  
3. `initial_minmax_ratio`  
4. `initial_nmadydt`  
5. `final_q50`  
6. `final_cv`  
7. `final_minmax_ratio`  
8. `final_nmadydt`  
9. `cv_total`  
10. `minmax_ratio_total`  
11. `nmadydt_total`  
12. `mean_total`  
13. `median_total`

> The RFC expects inputs in **this exact order** (see `rfc_metadata.json["features"]`). Any change requires retraining.

---
