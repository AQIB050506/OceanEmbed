# Architecture Decisions Log

## Project: OceanEmbed — Subsurface Ocean Temperature Reconstruction
**SIH Problem Statement ID:** SIH26066 | **Team:** Neural Shadows

---

## Decision #1: Primary Reconstruction Architecture
**Date:** 2026-09-13
**Decision:** Use Attention-Enhanced 3D U-Net++ (no two-stage transfer learning — single-stage training)

**Justification:** After reviewing 6+ prior-art papers (TS-Cast, DORS, attention-3D-UNet++, STGAT, ViT+clustering, SwinOcean3D), the attention-enhanced 3D U-Net++ offers the best tradeoff between reconstruction accuracy, hackathon feasibility, and alignment with our North Indian Ocean problem. The 3D convolutional structure naturally captures depth-dependent temperature structure, CBAM attention gates focus on depth-relevant features. The original plan included two-stage transfer learning (monthly Argo pretrain → daily GLORYS finetune) but was simplified to single-stage GLORYS training due to timeline constraints. This approach has been validated on the dynamically similar northwestern Pacific region (Wang et al., 2026).

---

## Decision #2: Input Variable Set
**Date:** 2026-09-13
**Decision:** 5 channels: SST + SSH/SLA + SSS + surface winds (U, V)

**Justification:** Literature consistently shows SST dominates mixed-layer reconstruction and SSH dominates thermocline/deep reconstruction (confirmed by SHAP analysis in Liu et al. 2026). SSS is important for Bay of Bengal freshwater dynamics. Winds improve upwelling region estimates. SST and SSH are available in near-real-time; SSS and winds have 1-7 day latency, acceptable for our training pipeline.

---

## Decision #3: Training Target Data
**Date:** 2026-09-13
**Decision:** Use GLORYS12V1 reanalysis as training label, with INCOIS Argo as independent validation

**Justification:** GLORYS12V1 is the gold-standard ocean reanalysis, assimilating real observations into a physical ocean model. It provides daily 3D fields at 1/12° resolution. INCOIS Argo is genuinely independent (not assimilated into GLORYS for recent years) and specifically designed for Indian Ocean validation. This separation ensures honest evaluation per rules.md §3.

---

## Decision #4: Temporal Range
**Date:** 2026-09-13
**Decision:** Training period: 2010–2025 (16 years)

**Justification:** SMOS SSS available from 2010, SMAP from 2015 (combined product starts 2010). Scatterometer winds from Metop available from 2007. OSTIA SST from 2006. DUACS SLA from 1993. The overlap window ensuring all inputs are available starts at 2010. This gives ~5,840 daily samples — sufficient for deep learning training while staying within storage budget.

---

## Decision #5: Spatial Domain & Resolution
**Date:** 2026-09-16 (updated to match downloaded data)
**Decision:** North Indian Ocean: **0°N–25°N, 45°E–100°E** at 0.25° × 0.25° daily; depth levels **0–450m** (15 levels)

**Justification:** The original PRD specified 5°N–30°N, 45°E–105°E, 0–1000m. However, the actual downloaded datasets (GLORYS regional subset, satellite products) cover 0–25°N, 45–100°E and GLORYS depth only extends to ~450m. Decision: use what was actually downloaded rather than re-downloading — this is acceptable for a hackathon PoC. Impact: domain is smaller and shallower than PRD spec; cannot reconstruct below 450m.

**Depth levels used:** [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 450] m

---

## Decision #6: Temporal Window
**Date:** 2026-09-17
**Decision:** temporal_window = 5 days (reduced from originally designed 26 days)

**Justification:** 26-day windows caused 25s/batch on the RTX 4050 (6GB VRAM), making 100-epoch training impractical (~weeks). Reducing to 5 days gave 0.3s/batch, enabling full training in ~19 hours. Trade-off: less temporal context, but validated results (Argo RMSE=1.02°C) confirm the model still performs well.

---

## Decision #7: Loss Function
**Date:** 2026-09-17
**Decision:** Combined OceanEmbedLoss = depth-weighted MSE (2× thermocline) + 0.1 × uncertainty-aware MSE

**Justification:** Thermocline (50–200m) is the hardest zone to reconstruct (steepest gradients) and where Argo validation shows weakest correlation. Depth weighting emphasizes this region. Uncertainty-aware MSE (from TS-Cast) learns per-depth error variance, improving calibration. Density constraint loss was implemented but disabled (requires predicted salinity, which the temperature-only model doesn't produce).

---

## Decision #8: Demo Architecture
**Date:** 2026-09-19
**Decision:** Two-mode demo: (a) FastAPI server with live inference + precomputed, (b) static HTTP server for pure offline fallback

**Justification:** Judging requires fully offline operation. FastAPI mode (`run.py demo`) provides live inference on user-entered surface data plus precomputed browsing — full features. Static mode (`serve_offline.py`) works without Python/torch installed, serves only precomputed data with graceful degradation of the Live tab. JS libraries (Leaflet, Chart.js) bundled locally. Map tiles attempt online load with automatic grid-background fallback when offline.
