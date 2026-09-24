# Technical Architecture Document

## Project: OceanEmbed — Subsurface Ocean Temperature Reconstruction
**SIH Problem Statement ID:** SIH26066 | **Team:** Neural Shadows

---

## 1. System Overview

OceanEmbed is a pipeline that ingests daily surface satellite ocean observations, learns a compact latent representation of surface ocean state, and reconstructs subsurface temperature profiles at 15 standard depth levels — validated against real ARGO float measurements and surfaced through a demo application.

At a high level, data flows through five stages:

```
[Raw Satellite Data] → [Harmonization Pipeline] → [Embedding Model] →
[Reconstruction Model] → [Validation + Application Layer] → [Demo UI]
```

---

## 2. High-Level Architecture Diagram (textual)

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES LAYER                        │
│  SST | SSS | SSH/SLA | Surface Currents (U,V) | Surface Winds     │
│  (multiple satellite products, native resolutions/formats)       │
│  Target: GLORYS Reanalysis (temperature) | Gridded ARGO (val.)   │
└───────────────────────────┬────────────────────────────────────--┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING & HARMONIZATION                   │
│  - Format standardization (NetCDF ingestion)                     │
│  - Spatial regridding → 0.25° × 0.25°                             │
│  - Temporal alignment → daily resolution                         │
│  - Domain crop → North Indian Ocean (5°N–30°N, 45°E–105°E)       │
│  - Missing-data interpolation / gap filling (cloud cover, etc.)  │
│  - Normalization / scaling per variable                          │
└───────────────────────────┬────────────────────────────────────--┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│                    EMBEDDING GENERATION LAYER                    │
│  Input: stacked multi-channel surface grid (SST,SSS,SSH,U,V,...) │
│  Model: CNN / ViT / Autoencoder encoder                          │
│  Output: compact latent representation per grid cell/patch       │
│  (captures nonlinear ocean dynamics: eddies, thermocline shifts) │
└───────────────────────────┬────────────────────────────────────--┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│                  RECONSTRUCTION MODEL LAYER                      │
│  Input: latent embedding (+ optional lat/lon, day-of-year)       │
│  Model: Decoder / attention-hybrid / 3D U-Net-style head         │
│  Output: Temperature at 15 standard depths (0–1000m) per cell    │
└───────────────────────────┬────────────────────────────────────--┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│                   VALIDATION & EVALUATION LAYER                  │
│  - Compare reconstructed profiles vs. held-out ARGO profiles     │
│  - Metrics: Correlation, RMSE, Bias (per depth level)            │
│  - Error analysis: depth-wise, seasonal, regional breakdown      │
└───────────────────────────┬────────────────────────────────────--┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│  - Ocean Heat Content indicator (cyclone-relevant)               │
│  - Marine heatwave / thermal anomaly flagging                    │
│  - Depth-profile query at any lat/lon                            │
└───────────────────────────┬────────────────────────────────────--┘
                            ↓
┌──────────────────────────────────────────────────────────────────┐
│                       DEMO / UI LAYER                            │
│  - Interactive map (Bay of Bengal / Arabian Sea)                 │
│  - Click a location → live-generated temp-vs-depth profile       │
│  - Overlay: reconstructed profile vs. real ARGO reading          │
│  - Anomaly/heatwave map overlay                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Data Ingestion & Harmonization
- **Inputs:** SST, SSS, SSH/SLA, surface currents (U,V), surface winds (U,V) — each from its own satellite/reanalysis source, likely differing in native resolution and file format (mostly NetCDF).
- **Target/label data:** GLORYS reanalysis temperature (training), Gridded ARGO (independent validation).
- **Key operations:**
  - Regridding all inputs to a common 0.25° × 0.25° grid over the North Indian Ocean domain.
  - Temporal resampling to daily resolution.
  - Gap-filling for missing values (e.g., cloud-occluded SST pixels) — may use existing gap-filled satellite products where available.
  - Feature normalization per variable before model input.
- **Output:** A clean, aligned multi-channel daily surface dataset ready for model consumption.

### 3.2 Embedding Generation
- **Purpose:** Compress high-dimensional multi-variable surface fields into a compact latent representation that captures the nonlinear signatures of subsurface processes (thermocline displacement, mesoscale eddies).
- **Candidate approaches:**
  - CNN encoder (spatial feature extraction over local surface patches)
  - Vision Transformer (attention over broader spatial context)
  - Autoencoder (learned compressed representation, potentially pretrained self-supervised on surface data alone)
- **Design consideration:** architecture choice should be justified against the physical reasoning of the signal (e.g., CNNs for local spatial patterns like eddies, attention/ViT for longer-range dependencies).

### 3.3 Reconstruction Model
- **Purpose:** Map the latent embedding (optionally combined with location and time-of-year features) to a full 15-depth temperature profile.
- **Candidate approaches:**
  - Decoder head (mirrors encoder, autoencoder-style)
  - Attention-based hybrid architecture (e.g., cross-attention between surface embedding and depth queries)
  - 3D U-Net style architecture (as used in comparable published work) for joint spatial + depth structure learning
- **Output shape:** For each grid cell, a vector of 15 temperature values corresponding to standard depth levels (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m).

### 3.4 Validation Framework
- **Method:** Hold out a subset of real ARGO float profiles (not used in training) as ground truth.
- **Metrics per depth level:**
  - Correlation coefficient
  - RMSE
  - Bias (systematic over/underestimation)
- **Reporting:** Depth-wise and seasonal/regional breakdowns to identify where the model performs well vs. poorly (e.g., near-surface likely more accurate than deep layers).

### 3.5 Application Layer
- Translates raw reconstructed fields into decision-relevant signals:
  - **Ocean Heat Content (OHC) estimate** — relevant to cyclone intensification risk.
  - **Marine heatwave / thermal anomaly detection** — flags abnormal subsurface warming relative to climatology.
- This layer is what converts a research-grade reconstruction into something INCOIS or a judge can immediately see the value of.

### 3.6 Demo / UI Layer
- Interactive map of the Bay of Bengal / Arabian Sea domain.
- User selects a location → system displays a live-generated temperature-vs-depth profile.
- Side-by-side comparison: model-reconstructed profile vs. actual ARGO measurement at that location (for validation locations).
- Overlay showing flagged anomaly/heatwave regions across the basin.

---

## 4. Data Flow Summary

1. Raw satellite + reanalysis data ingested (NetCDF).
2. Harmonization pipeline standardizes to common grid/resolution/domain.
3. Embedding model compresses surface state into latent vectors.
4. Reconstruction model predicts 15-depth temperature profile per grid cell/day.
5. Validation module scores predictions against held-out ARGO data.
6. Application layer derives OHC/anomaly signals from reconstructed fields.
7. Demo UI renders interactive visualizations for presentation.

---

## 5. Technology Stack (proposed — to be finalized)

| Layer | Likely Tools |
|---|---|
| Data handling | Python, xarray, netCDF4, NumPy, Pandas |
| Regridding/interpolation | xESMF / scipy interpolation |
| Deep learning | PyTorch or TensorFlow |
| Model architectures | CNN, ViT, Autoencoder, Attention-based hybrid (final choice pending literature-informed decision) |
| Validation/analysis | scikit-learn metrics, Matplotlib/Seaborn for error analysis |
| Demo backend | Flask / FastAPI (serving model inference) |
| Demo frontend | Web map interface (e.g., Leaflet/Mapbox) + chart library for depth-profile plots |
| Compute | GPU access required for training (cloud or local — TBD with mentor) |

---

## 6. Key Architectural Risks

| Risk | Mitigation |
|---|---|
| Inconsistent native resolutions/formats across satellite sources | Build harmonization pipeline early and validate on a small spatial/temporal subset before scaling up |
| Missing data (cloud cover, sensor gaps) | Use existing gap-filled satellite products where possible; fall back to interpolation |
| Model overfitting to training region/season | Hold out ARGO profiles across multiple seasons/years for validation, not just a random split |
| Demo not visually compelling | Prioritize the profile-comparison and anomaly-map UI early, not as an afterthought |
| Compute constraints in hackathon timeframe | Consider training on a smaller spatial subset first (e.g., Bay of Bengal only) before attempting full domain |

---

## 7. Open Architectural Decisions (to finalize with mentor)

- Final choice of embedding architecture (CNN vs. ViT vs. Autoencoder) and justification.
- Whether reconstruction is done as point-to-point, surface-patch-to-point, or full 3D volumetric prediction (see prior-art approaches).
- Which specific satellite products to standardize on for each input variable.
- Compute environment (local GPU vs. cloud) and expected training time budget.
- Whether the applied output layer is built as part of the core deliverable or added as a stretch goal if time permits.
