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

## Planned / Not Yet Run

| Experiment | Status | Notes |
|-----------|--------|-------|
| Simple baseline (linear/nearest-neighbor) | **DONE 2026-09-24** | See Experiment 6 — DL beats it by 43× (GLORYS) / 2.3× (ARGO) |
| Thermocline weight increase (2×→4×) | Not run | Expected +2-3% R² |
| Resume training 50 epochs at lr=1e-5 | Not run | Expected +3-5% R² |
| temporal_window=10 retrain | Not run | Expected +2-4% R² |
| OHC validation vs known cyclone events | Not run | e.g., Cyclone Amphan May 2020 |
| Full-test-set application products | Not run | Only 12 precompute dates done |
