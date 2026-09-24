# Product Requirements Document (PRD)

## Project: OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction

**SIH Problem Statement ID:** SIH26066
**Organization:** Ministry of Earth Sciences (MoES)
**Department:** Indian National Centre for Ocean Information Services (INCOIS) — Ocean Valley
**Category:** Software
**Theme:** Space Technology
**Team:** Neural Shadows
**Event:** Smart India Hackathon 2026

---

## 1. Problem Statement

Subsurface ocean temperature is a fundamental variable for understanding ocean circulation, upper-ocean heat content, stratification, air-sea interaction, and marine ecosystems. It underpins critical applications such as marine heatwave monitoring, fisheries management, cyclone intensity forecasting, and climate data assimilation.

Direct subsurface measurements are collected primarily via ARGO profiling floats, moored buoys, gliders, and ships — but this network is sparse (roughly one float every few hundred kilometers, updated every ~10 days) and cannot provide continuous, basin-scale coverage.

In contrast, satellites continuously observe the ocean *surface* — Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Sea Surface Height/Sea Level Anomaly (SSH/SLA), surface currents, and surface winds — at high spatial and temporal resolution. Surface variables carry indirect signatures of subsurface processes (thermocline displacement, mesoscale eddies, vertical mixing) through known physical mechanisms, but the relationship is nonlinear and not directly observable without a subsurface measurement.

**Core need:** A system that reconstructs the 3D subsurface temperature structure of the North Indian Ocean using only surface satellite observations, filling the gap left by sparse in-situ sampling.

---

## 2. Objective

Develop a deep learning framework that:
- Learns the nonlinear mapping between surface satellite observations and subsurface temperature profiles.
- Reconstructs daily subsurface temperature fields at 15 standard depth levels (0–1000 m) across the North Indian Ocean (5°N–30°N, 45°E–105°E).
- Operates at 0.25° × 0.25° spatial resolution and daily temporal resolution.
- Validates reconstruction accuracy against independent ARGO observations.
- Demonstrates a working Proof-of-Concept over the Bay of Bengal / Arabian Sea.

---

## 3. Scope

### In Scope
- Data preprocessing and harmonization pipeline for multi-source satellite and ocean datasets.
- Deep learning model(s) generating satellite embeddings and reconstructing subsurface temperature.
- Validation framework against independent ARGO profiles using correlation, RMSE, and bias metrics.
- A demo interface visualizing reconstructed temperature profiles at selectable locations, compared against real ARGO readings.
- An applied "so-what" layer on top of raw reconstruction — e.g., marine heatwave flagging or a cyclone-relevant ocean heat content indicator — to make the output actionable, not just a metric.

### Out of Scope (for hackathon PoC)
- Global ocean coverage (restricted to North Indian Ocean domain per problem statement).
- Real-time operational deployment/integration into INCOIS production systems.
- Salinity or current reconstruction (temperature only, per problem statement).

---

## 4. Input Data

| Variable | Source (recommended) |
|---|---|
| Sea Surface Temperature (SST) | Satellite (to be finalized — e.g., OSTIA/MUR) |
| Sea Surface Salinity (SSS) | Satellite (e.g., SMAP/SMOS-derived) |
| Sea Surface Height / Sea Level Anomaly (SSH/SLA) | Satellite altimetry |
| Surface Currents (U, V) | Satellite-derived / reanalysis |
| Surface Winds (U, V) | Satellite scatterometer / reanalysis |

All inputs standardized to 0.25° spatial resolution, daily temporal resolution. Where a dataset isn't natively available at this resolution, spatial/temporal interpolation or regridding will be applied.

## 5. Target (Training Label) Data

- **GLORYS Global Ocean Reanalysis** — Temperature (https://doi.org/10.48670/moi-00021)
- **Gridded ARGO** — INCOIS Live Access Server (LAS), used as independent validation set

---

## 6. Proposed Technical Approach

### 6.1 Pipeline Stages
1. **Data harmonization** — ingest multi-source satellite products, regrid to common 0.25°/daily grid.
2. **Embedding generation** — compress multidimensional surface observations into a compact latent representation capturing hidden ocean dynamics.
3. **Reconstruction model** — learn mapping from surface embedding → temperature profile at 15 depth levels.
4. **Validation** — compare reconstructed profiles against held-out ARGO observations using correlation, RMSE, bias.
5. **Application layer** — surface an interpretable output (e.g., anomaly/heatwave flag, ocean heat content estimate) rather than raw numbers only.

### 6.2 Candidate Architectures (to be finalized after literature review)
- Convolutional Neural Networks (CNN)
- Vision Transformers (ViT)
- Autoencoders
- Graph Neural Networks (GNN)
- Attention-based hybrid architectures (e.g., CNN + attention, 3D U-Net with attention)

### 6.3 Standard Depth Levels (meters)
0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000

---

## 7. Related Work / Prior Art

This problem area is an active global research field, not a novel concept:
- **TS-Cast (2026)** — uncertainty-aware deep learning for subsurface reconstruction, northwestern Pacific.
- **North Atlantic study (2026)** — interpretable deep learning framework using SST, SSH, SLA, surface wind, SST gradient (near-identical input set to this problem statement).
- **DORS global product** — ConvLSTM-based global subsurface temperature reconstruction (upper 2000m, 1993–2020), combining satellite and ARGO data.
- **Attention-enhanced 3D U-Net++ (2026)** — northwestern Pacific, daily T-S fields at 0.25° resolution, 5–2000m depth, using SST + SSH.
- **Global ViT-based framework (2026)** — Vision Transformer + CNN + Attention U-Net ensemble for subsurface reconstruction.

**Implication:** The core method (surface-to-subsurface deep learning reconstruction) is established in literature. Differentiation for this project will come from:
- A India-regional focus (North Indian Ocean specifically), validated against regional ARGO data.
- Building a usable, applied output layer (e.g., cyclone-relevant heat content indicator or heatwave early-warning) on top of the reconstruction, rather than stopping at accuracy metrics.
- Clean execution and demo quality, since the underlying method is de-risked by prior published work.

---

## 8. Success Metrics

| Metric | Purpose |
|---|---|
| Correlation (reconstructed vs. ARGO) | Overall skill of the model |
| RMSE | Magnitude of reconstruction error |
| Bias | Systematic over/under-estimation |
| Depth-wise accuracy breakdown | Identify where model performs best/worst (surface vs. deep layers) |
| Qualitative demo comparison | Reconstructed profile vs. real ARGO profile at a held-out location, shown live |

---

## 9. Expected Deliverables (per problem statement)

- End-to-end preprocessing pipeline for satellite and ocean datasets.
- Satellite embedding engine capable of learning latent ocean representations.
- Deep learning reconstruction model for subsurface temperature.
- Standardized output at daily temporal resolution, 0.25° spatial resolution.
- Validation framework using independent ARGO observations.
- Working Proof-of-Concept demonstrated over the Bay of Bengal / Arabian Sea.

---

## 10. Real-World Impact

- **Cyclone intensity forecasting** — subsurface ocean heat content is a key driver of rapid cyclone intensification; better subsurface data improves forecast accuracy and evacuation lead time.
- **Marine heatwave / coral bleaching early-warning** — earlier detection of abnormal subsurface warming.
- **Fisheries guidance** — temperature-layer data helps predict fish concentration zones.
- **Cost-effective ocean monitoring** — supplements the sparse, expensive ARGO float network using existing free satellite data.
- **Climate science** — supports long-term ocean heat content tracking for coastal and climate policy planning.

---

## 11. Key Risks

| Risk | Notes |
|---|---|
| Data harmonization complexity | Multiple satellite sources with different native resolutions/grids — likely the most time-consuming part of the build. |
| Timeline compression | This is a multi-month research-grade problem; hackathon timeline requires tight scoping of pipeline vs. modeling effort. |
| Demo storytelling | Raw reconstruction accuracy doesn't visually "wow" — needs translation into a live, visual, comparison-based demo (reconstructed vs. real ARGO profile; heatwave flag). |
| Domain knowledge gap | Team's AI/DS background is strong, but physical oceanography intuition (thermocline dynamics, eddy signatures) may need to be learned quickly via literature review. |
| Architecture selection risk | Multiple valid architecture options exist; needs a clearly reasoned choice (not "we used CNN because it's standard"), since judges may probe this. |

---

## 12. Open Questions (to resolve with mentor)

- Which specific satellite SST/SSS/SSH/wind products should we standardize on for the training pipeline?
- What compute resources are available for training (GPU access, cloud credits)?
- Should the applied output layer prioritize cyclone heat-content indication or marine heatwave detection as the primary demo hook?
- What is the team's realistic timeline split between data pipeline work vs. model development vs. demo/UI polish?
