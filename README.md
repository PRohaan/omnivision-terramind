# Omnivision — TerraMind Multi‑Modal Earth Observation Studio (Generation + Evaluation)

**Omnivision** is an interactive Earth‑observation “lab” built on the **TerraMind** geospatial foundation model (via **TerraTorch**).  
It showcases TerraMind’s **any‑to‑any generation** across key modalities and adds **ground truth comparison, per‑tile metrics, and interpretable error maps**, so results are measurable—not just qualitative.

**Team:** Omnivision  
**Contacts:** rohaanrasool110@gmail.com • tamseelfatimah245@gmail.com • solehsafida@gmail.com  
**Demo video:** https://www.youtube.com/watch?v=yhdLLpik4a0  
**Challenge:** IBM/ESA Geospatial — BlueSky Challenge (Hugging Face Community Submission)

---

## Why it matters
Earth observation workflows increasingly require multi‑sensor data (Optical, SAR, DEM, vegetation indices, land cover). However, working across these modalities is slow and fragmented—different formats, different preprocessing, and limited tools to both *generate* and *validate* derived layers.

**Omnivision** provides a modern, reproducible interface to explore TerraMind’s multi‑modal generation and to **evaluate outputs against ground truth** on the provided example tiles.

---

## What Omnivision does
### Core capabilities
- **Any‑to‑any generation** with TerraMind (select input modality → generate multiple output modalities)
- **Examples mode**: automatically loads matching **Ground Truth** rasters when available (same file id across modalities)
- **Side‑by‑side comparison**: **Input | Generated | Ground Truth**
- **Per‑tile evaluation metrics**
  - Continuous modalities: **MAE, RMSE, Pearson r, SAM**
  - Categorical (LULC): **Overall Accuracy, Macro‑F1**
- **Error visualization** (mismatch map) with an on‑screen legend (low→high error)
- **Download outputs** (PNG visualization) directly from the UI
- **Animated background fallback** (if no local background image is present)

---

## Supported modalities (as in TerraMind notebooks)
- **S2L2A** — Sentinel‑2 Optical (multi‑spectral)
- **S1RTC / S1GRD** — Sentinel‑1 SAR (VV/VH)
- **DEM** — Elevation
- **NDVI** — Vegetation index
- **LULC** — Land Use / Land Cover

---

## Screenshots
> Add these files to `assets/` (already included in this repository).

![Landing](assets/landing_page.png)  
![Comparison](assets/comparison.png)  
![Metrics](assets/metrics.png)
![Visual Error](assets/visual_error.png)
![Lagend](assets/lagend.png)

---

## How it works (technical summary)
Omnivision uses TerraTorch’s model registry to build TerraMind generation models:

```python
FULL_MODEL_REGISTRY.build(
  "terramind_v1_base_generate",
  modalities=[input_modality],
  output_modalities=[...],
  pretrained=True,
  standardize=True
)
