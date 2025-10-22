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
