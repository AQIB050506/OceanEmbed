# Experiment Log

## Project: OceanEmbed — Subsurface Ocean Temperature Reconstruction
**SIH Problem Statement ID:** SIH26066 | **Team:** Neural Shadows

Per `rules.md` §3: every architecture/hyperparameter/metrics run gets logged here.

---

## Experiment 1 — Test Training (Sanity Check)
**Date:** 2026-09-17
**Script:** `train_model.py`
**Config:** 3 epochs, batch_size=2, temporal_window=5, lr=1e-4
**Purpose:** Verify training pipeline works end-to-end before committing to full run
**Result:** PASS — forward/backward pass works, loss decreases, checkpoints save

---

## Experiment 2 — Full Training Run
**Date:** 2026-09-17 → 2026-09-19
**Script:** `train_full.py`
**Config:**
| Parameter | Value |
|-----------|-------|
| Architecture | AttentionUNet3D (3D U-Net++ + CBAM) |
| Parameters | 9,791,217 |
| Batch size | 2 |
| Temporal window | 5 days |
| Learning rate | 1e-4 |
| Optimizer | AdamW (weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Epochs | 100 |
| Gradient clipping | 1.0 |
| Device | NVIDIA RTX 4050 Laptop (6.4 GB VRAM) |
| Loss | OceanEmbedLoss (depth-weighted MSE + 0.1×uncertainty MSE) |

**Training Progress:**
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.1230 | 0.0276 |
| 50 | 0.0050 | 0.0100 |
| 98 | 0.0037 | 0.0096 |
| 100 | 0.0034 | 0.0096 |
| **Best (96)** | — | **0.009565** |

**Duration:** ~19 hours (114s/epoch × 100 epochs)
**Checkpoint:** `models/best_model.pt` (112 MB)
**Result:** PASS — no overfitting, converges steadily

---

## Experiment 3 — GLORYS Test Set Validation
**Date:** 2026-09-19
**Script:** `validate_test.py`
**Data:** Test split (2024-01-01 → 2025-12-31), 346 samples, compared to held-out GLORYS targets
**Result:**

| Metric | Value |
|--------|-------|
| Overall RMSE | 0.083°C |
| Overall MAE | 0.045°C |
| Overall R² | 0.989 |
| Surface (0-30m) R² | > 0.99 |
| Thermocline (100-150m) R² | 0.80–0.90 |
| Deep (200-450m) R² | > 0.99 |

**Artifacts:** `models/validation_results/test_metrics.json`, `test_predictions.nc`
**Result:** PASS

---

## Experiment 4 — INCOIS Argo Independent Validation
**Date:** 2026-09-19
**Script:** `validate_argo.py`
**Data:** Raw INCOIS Argo float observations (364 MB, 8.8M rows), filtered to domain + QC + test-date overlap
**Comparisons:** 56,925 point comparisons, 345 common dates, 15 depth levels
**Result:**

| Metric | Value |
|--------|-------|
| Overall RMSE | 1.02°C |
| Overall Correlation | 0.83 |
| Surface (0-20m) RMSE | < 0.75°C, Corr > 0.92 |
| Thermocline (100m) RMSE | 2.05°C (weakest) |
| Deep (250-450m) RMSE | < 0.9°C, Corr 0.76–0.88 |

**Artifacts:** `models/validation_results/argo_validation_metrics.json`
**Result:** PASS — competitive with published methods for independent Argo validation

---

## Experiment 5 — Argo Gridding (GPU-accelerated)
**Date:** 2026-09-18
**Script:** `src/preprocessing/process_argo.py`
**Config:** Chunk size=100 dates, depth tolerance=10m, PyTorch CUDA scatter-add binning
**Result:** 3.7M observations → 421,649 valid grid cells across 30,809 dates; 41.3 GB output; ~2.5 min processing time
**Note:** Gridded file produced but `validate_argo.py` bypasses it (uses raw observations directly for point-matching)

---

## Experiment 6 — Demo Precompute
**Date:** 2026-09-19 (initial), 2026-09-24 (with ARGO + anomaly)
**Script:** `demo/precompute_demo.py`
**Config:** 12 dates (2024 monthly), 12 locations across North Indian Ocean
**Outputs per date:** 12 profiles, SST map (101×221), OHC map, MHW flags, cyclone risk, anomaly map, ARGO comparison profiles
**Artifact:** `demo/frontend/data/demo_data.json` (~10 MB)
**Result:** PASS

---

## Known Issues / Failed Attempts

| Issue | Resolution |
|-------|-----------|
| batch_size=8 OOMs on 6GB VRAM | Reduced to 2 |
| temporal_window=26 too slow (25s/batch) | Reduced to 5 (0.3s/batch) |
| Trainer hardcoded `torch.device("cuda")` | Fixed to respect `device` parameter |
| `OceanEmbedLoss.depth_weights` buffer on CPU | Added `.to(DEVICE)` after criterion creation |
| INCOIS Gridded Argo server down during competition | Built `validate_argo.py` to use raw float data directly |

---

## Experiment 6 — Baseline Models (Phase 2)
**Date:** 2026-09-24
**Script:** `baseline.py`
**Config:** Linear regression (per-depth OLS on [sst, ssh, sss, wind_u, wind_v, 1]) + climatology (train-set mean profile); fit on 1095/4380 train days (subsample=4); evaluated on full 346-sample test set + raw ARGO (56,925 points, 345 dates)

**Results:**

| Model | GLORYS RMSE | GLORYS R² | ARGO RMSE | ARGO Corr |
|-------|------------|-----------|-----------|-----------|
| Climatology | 3.72°C | 0.59 | 2.79°C | — |
| Linear regression | 3.55°C | 0.62 | 2.36°C | 0.54 |
| **DL (AttentionUNet3D)** | **0.083°C** | **0.989** | **1.02°C** | **0.83** |

**Improvement:** DL beats linear baseline by **43× on GLORYS RMSE**, **2.3× on ARGO RMSE**.

**Artifacts:** `models/validation_results/baseline_metrics.json`, `baseline_params.npz`
**Result:** PASS — baseline established; DL improvement honestly quantified per rules.md §3

---

## Experiment 7 — Online Demo Real-Time Pipeline
**Date:** 2026-09-24
**Script:** `demo/online/realtime_fetcher.py`, `demo/online/server.py`
**Config:** Open-Meteo Marine (SST+SSH) + Weather (wind), 66-point sparse grid, 12-thread parallel fetch, cubic interpolation to 101×221; model input (5,5,101,221) with training norm stats (Kelvin SST)
**Result:** Fetch ~12s, GPU inference 0.49s, flat-profile cells 30.8%→0.3% after switching SSH from random to API-sourced; all endpoints verified 200; realistic profiles (e.g. Arabian Sea 28.4→11.9°C)

**Artifacts:** `demo/online/frontend/index.html` (overlays + legend), server endpoints `/api/realtime`, `/api/profile`, `/api/sst_map`, `/api/pred_map`
**Result:** PASS

---

## Experiment 8 — 3D WebGL Frontend (Three.js)
**Date:** 2026-09-24
**Stack:** Three.js r160 (vendored, importmap, no build step) + OrbitControls + Chart.js
**Features:** wave-shader SST surface (vertex displacement), 15 stacked depth planes, depth-slicing slider (0–450m, lerped opacity), raycast click→profile, pop marker, domain wireframe/depth labels/fog, count-up indicators, staggered card reveals
**API:** new `/api/volume?stride=2` — 15 layers × 51×111 grid (~5661 values/layer), ~113KB/layer JSON; full-res `land` mask (101×221) + `land_h/land_w`
**Geography layer (24 Sep, same day):** land mask rendered into textures (dark earth + bright coastline cells), wave suppression over land (vertex-shader mask sampling), 5° graticule + coordinate ticks baked into surface texture, 11 billboard region labels placed via land-mask verification (Maldives=land/atolls), Leaflet minimap (Esri dark-gray keyless tiles — CartoDB requires API key now, returns placeholder), two-way selection sync (`selectPoint`), land-click rejection
**Slider↔chart sync (25 Sep):** `highlightDepth()` resizes/recolors the active point (7px yellow) + custom Chart.js plugin draws dashed crosshair, depth pill, and °C label at the active row; hooked into `applyDepthSlice` so slider drags update instantly (`update('none')`); y-axis flipped to surface-at-top (oceanographic convention, matches slider direction)
**Fish Forecast port (25 Sep):** FISH_DB (21 species: 9 surface / 7 mid / 5 deep) + `matchFishes` (best temp within species depth range, must fall inside temp range) ported from offline demo; side-panel card with FISHERMAN AID badge, zone-grouped species cards, habitat scatter chart (per-species temp×depth segments colored by zone + amber water profile); fish chart also draws the slider crosshair (`fishDepthHL` at `DEPTHS[idx]`); hook: `fetchProfile → renderFishForecast`
**Fish icons revamp (25 Sep):** all emoji (21 species icons + zone labels) replaced per decisions #20 — Lucide `fish`/`shrimp`/`shell` SVGs inlined (`FISH_ICONS` map, ISC), icon tinted by zone, cards gain 3px zone `border-left` accent, zone labels gain 7px dot, empty states reuse muted `fish` glyph; species→icon: 19 fish, 1 shrimp, 1 shell (no squid glyph exists in Lucide/Tabler)
**Verification:** `node --check` on extracted module JS; headless Edge screenshots (initial + fully-loaded states); minimap tile load verified by pixel analysis (3073 unique colors, row-structure std 11.5, 0.49% bright = no API-KEY placeholder); chart highlight verified by pixel analysis (crosshair/pill/point zones: 37/215/74 yellow px at 0m row); fish chart verified by pixel analysis (amber profile surface point at bbox top, crosshair at 0m row, 10 card-border rows + 351 badge px); fish icons verified by pixel analysis on 1600×2700 shot (3px accent runs at x1209: 9 cyan / 7 blue / 4 indigo cards; 14px icon clusters per card in all 3 zones; label dots: cyan@y1060, blue@y1566, indigo@y1966 exact zone colors); 0 emoji codepoints in index.html; all endpoints 200; pytest 5/5
**Result:** PASS — 3D scene renders with land/labels/graticule, default profile auto-loads (SST 28.9°C, thermocline 124m), minimap shows domain + synced marker

---

## Planned / Not Yet Run

| Experiment | Status | Notes |
|-----------|--------|-------|
| Simple baseline (linear/nearest-neighbor) | **DONE 2026-09-24** | See Experiment 6 — DL beats it by 43× (GLORYS) / 2.3× (ARGO) |
| Thermocline weight increase (2×→4×) | Not run | Expected +2-3% R² |
| Resume training 50 epochs at lr=1e-5 | Not run | Expected +3-5% R² |
| temporal_window=10 retrain | Not run | Expected +2-4% R² |
| OHC validation vs known cyclone events | Not run | e.g., Cyclone Amphan May 2020 |
| Full-test-set application products | Not run | Only 12 precompute dates done |
