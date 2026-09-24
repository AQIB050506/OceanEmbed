# PROJECT STATUS — OceanEmbed

> **Last updated:** 24 Sep 2026 — after offline demo fixes (map fallback, ARGO overlay, map layers).
> **Read this file first** when resuming work. It tells the new session exactly where we are.

---

## 1. Project Overview

**OceanEmbed** is a deep learning framework for reconstructing 3D subsurface ocean temperature profiles in the North Indian Ocean from satellite surface observations. Built for SIH 2026 (Problem ID: SIH26066) by Team Neural Shadows.

- **Architecture:** Attention-Enhanced 3D U-Net++ with CBAM attention gates
- **Training target:** GLORYS12V1 reanalysis
- **Validation:** INCOIS Argo (independent, raw float point-matching)
- **GPU:** NVIDIA RTX 4050 Laptop (6 GB VRAM)
- **Domain:** North Indian Ocean, 0–25°N, 45–100°E
- **Depth levels:** 15 levels, 0–450m: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 450]
- **Spatial resolution:** 0.25° × 0.25° daily (101 × 221 grid cells)

---

## 2. What Has Been Completed

### 2.1 Environment Setup
- [x] Python 3.12, PyTorch 2.5.1+cu121 (CUDA only)
- [x] All dependencies installed (requirements.txt)
- [x] pytest — all 5 tests pass

### 2.2 Data Downloads — ALL COMPLETE
- [x] GLORYS (17 yearly files 2010–2026), SST, SSH/SLA, SSS, Winds, Currents

### 2.3 Preprocessing — ALL COMPLETE
- [x] Harmonization → 0.25° grid, domain crop, 5-channel surface_combined.nc
- [x] Train/Val/Test splits: 4384 / 731 / ~730 timesteps
- [x] Normalization stats from training set
- [x] INCOIS Argo raw data downloaded (364 MB) + gridded (41.3 GB, GPU-accelerated)

### 2.4 Model Architecture (src/reconstruction/model.py)
- [x] AttentionUNet3D — 3D U-Net++ + CBAM + FiLM, ~9.8M params, 0.3 GB VRAM

### 2.5 Training — 100 EPOCHS COMPLETE
- [x] Best val loss: 0.0096 (epoch 96), saved as `models/best_model.pt` (112 MB)
- [x] Checkpoints every 10 epochs, training logs in `models/logs/`
- [x] 5 bug fixes applied during training (device handling, OOM, temporal window)

### 2.6 Validation — BOTH DONE
- [x] **GLORYS test set:** RMSE=0.083°C, R²=0.989 → `models/validation_results/test_metrics.json`
- [x] **INCOIS Argo (independent):** RMSE=1.02°C, Corr=0.83 → `models/validation_results/argo_validation_metrics.json`
  - Surface (0-20m): RMSE < 0.75°C, Corr > 0.92
  - Thermocline (100m): RMSE 2.05°C (weakest — expected for transition zone)
  - Deep (250-450m): RMSE < 0.9°C, Corr 0.76–0.88

### 2.7 Application Layer (src/application/products.py)
- [x] OHC calculation, marine heatwave detection, cyclone risk classification

### 2.8 Offline Demo — FIXED & ENHANCED (24 Sep 2026)
- [x] FastAPI backend (`demo/backend/app.py`) — live inference + precomputed
- [x] Frontend (`demo/frontend/index.html`) — dark ocean dashboard, Leaflet map, Chart.js
- [x] **Vendor libs bundled locally** (Leaflet, Chart.js) — no CDN dependency
- [x] **Map offline fallback** — tries Esri tiles, auto-switches to lat/lon grid background if offline
- [x] **API graceful degradation** — static server disables Live tab with clear message
- [x] **Map overlays** — SST / OHC / Anomaly layer toggle (rendered from precomputed grids)
- [x] **ARGO overlay** — reconstructed vs real ARGO dashed line + point-wise RMSE badge
- [x] **Fish forecast** — species matching by depth+temperature zones
- [x] **Cyclone presets** — Amphan, Fani, Biparjoy, Michaung, Tauktae one-click loading
- [x] Precomputed data: 12 dates × 12 locations with ARGO + anomaly maps → `demo/frontend/data/demo_data.json` (~10 MB)
- [x] Two server modes:
  - `python run.py demo` — full FastAPI (live inference + precomputed)
  - `python demo/serve_offline.py` — static fallback (precomputed only, no Python deps needed)

### 2.9 Online Demo (demo/online/) — BUILT & VERIFIED (24 Sep 2026)
- [x] Realtime fetcher — Open-Meteo Marine (SST+SSH) + Weather (wind), parallel ThreadPool (12 workers, ~12s)
- [x] Correct model input shape `(5, 5, 101, 221)` + training normalization stats (SST Kelvin conversion)
- [x] Real SSH from API (not random) — fixed 30.8% flat-profile collapse → 0.3%
- [x] Smooth climatological SSS (Arabian Sea salty / Bay of Bengal fresh)
- [x] Flat-profile climatology fallback in `/api/profile` (belt-and-suspenders)
- [x] Server: `/api/realtime`, `/api/profile`, `/api/sst_map`, `/api/pred_map?depth_idx=N`
- [x] Frontend: map overlays (None / Input SST / Predicted T @ depth selector), color legend, profile chart, indicators, validation stats card
- [x] E2E verified: inference 0.49s, realistic profile (28.4→11.9°C), all endpoints 200
- [x] Start: `python demo/online/start.py` or `python demo/online/server.py`

### 2.10 Tests & Documentation
- [x] 5 pytest tests passing (forward, backward, loss, spatial sizes, app products)
- [x] PRD.md, architecture.md, design.md, phase.md, rules.md, SETUP.md
- [x] MODEL_REPORT.md — full model report with metrics tables
- [x] docs/research-summary.md, docs/architecture-decisions.md (updated), docs/data-sources.md
- [x] **experiments.md** — complete experiment log (created 24 Sep)
- [x] **docs/decisions.md** — 13-entry decision log (created 24 Sep)

---

## 3. What Is NOT Done (Remaining Work)

### 3.1 Baseline Model (Phase 2) — DONE
- [x] Linear regression baseline: GLORYS RMSE=3.55°C, R²=0.62; ARGO RMSE=2.36°C, Corr=0.54
- [x] Climatology baseline: GLORYS RMSE=3.72°C, R²=0.59; ARGO RMSE=2.79°C
- [x] DL model beats baseline by 43× (GLORYS) / 2.3× (ARGO) — see `experiments.md` Exp. 6
- [x] Saved to `models/validation_results/baseline_metrics.json`

### 3.2 Application Layer on Full Test Set (Phase 5)
- [ ] Run OHC/MHW/cyclone risk on full test set (currently only 12 precompute dates)
- [ ] Validate OHC against known cyclone events (e.g., Cyclone Amphan May 2020)

### 3.3 Demo End-to-End Test (Phase 6)
- [ ] Run `python run.py demo`, verify both tabs, overlays, ARGO comparison
- [ ] Run `python demo/serve_offline.py`, verify static mode degradation
- [ ] Test fully offline (disconnect internet, verify grid fallback + precomputed works)
- [ ] Rehearse timed demo run (Phase 7)
- [ ] Record fallback screen capture (Phase 7)

### 3.4 Notebooks
- [ ] No Jupyter notebooks created yet (`notebooks/` empty)

### 3.5 Thermocline Improvement (stretch — MODEL_REPORT.md §8)
- [ ] Increase thermocline loss weight 2×→4× (+2-3% R² expected)
- [ ] Resume 50 epochs at lr=1e-5 (+3-5% R² expected)
- [ ] temporal_window 5→10 retrain (+2-4% R² expected)

### 3.6 Validation Plots
- [ ] Depth-wise RMSE/correlation charts (matplotlib) not generated — only JSON metrics exist

---

## 4. Known Issues & Decisions

### 4.1 Domain Mismatch — RESOLVED (see docs/decisions.md #5)
- config.py: 0–25°N, 45–100°E, 0–450m (matches downloaded data)
- PRD/architecture: 5–30°N, 45–105°E, 0–1000m (original spec)
- **Decision:** Use downloaded data; acceptable for hackathon PoC

### 4.2 Temporal Window — 5 days (not 26)
- 26 days = 25s/batch (weeks of training); 5 days = 0.3s/batch (19 hours)
- **Impact:** Less temporal context; Argo validation still achieves RMSE=1.02°C

### 4.3 Map Tiles Need Internet (mitigated)
- Esri dark tiles load from `server.arcgisonline.com`
- **Mitigation:** Auto-fallback to lat/lon grid background after 3s if tiles fail
- Precomputed overlays (SST/OHC/Anomaly) render locally regardless

### 4.4 Two Download Systems (historical)
- `download_*.py` scripts (copernicusmarine CLI) were used for actual downloads

---

## 5. What to Do in Next Session

### Step 1: Read this file (you're here)

### Step 2: ~~Build the baseline model~~ DONE
```bash
python baseline.py   # linear + climatology baselines evaluated on GLORYS + ARGO
```

### Step 3: End-to-end demo test
```bash
python run.py demo          # Full mode — test both tabs
python demo/serve_offline.py  # Static mode — test degradation
# Disconnect internet, reload — verify grid fallback
```

### Step 4: ~~Regenerate demo data with ARGO~~ DONE
```bash
python demo/precompute_demo.py   # 34 ARGO comparisons + anomaly maps
```

### Step 5: ~~Online demo~~ DONE — see §2.9
```bash
python demo/online/start.py   # port 8000
```

### Step 6: (Optional) Thermocline improvement
```bash
python train_full.py   # Resumes from checkpoint
```

---

## 6. File Map

```
D:\OceanEmbed\
├── src/
│   ├── config.py                          # All constants, paths, hyperparams
│   ├── reconstruction/
│   │   ├── model.py                       # AttentionUNet3D architecture
│   │   └── train.py                       # Losses, Trainer, checkpointing
│   ├── preprocessing/
│   │   ├── harmonize.py                   # Regridding, cropping, alignment
│   │   ├── dataset.py                     # PyTorch Dataset/DataLoader
│   │   └── process_argo.py                # GPU-accelerated Argo gridding
│   ├── validation/
│   │   └── evaluate.py                    # ARGO validation, metrics
│   ├── application/
│   │   └── products.py                    # OHC, MHW, cyclone risk
│   └── utils/
│       └── visualization.py               # Plotting helpers
├── demo/
│   ├── backend/app.py                     # FastAPI server (live + precomputed)
│   ├── frontend/index.html                # Dashboard UI (map, overlays, ARGO)
│   ├── frontend/data/demo_data.json       # Precomputed outputs (~10 MB)
│   ├── frontend/vendor/                   # Leaflet + Chart.js (bundled)
│   ├── serve_offline.py                   # Static fallback server
│   ├── precompute_demo.py                 # Generates demo_data.json
│   └── online/                            # Realtime demo (Open-Meteo APIs)
├── tests/
│   └── test_model.py                      # 5 tests, all passing
├── data/                                  # All downloaded + processed data
├── models/
│   ├── best_model.pt                      # Best checkpoint (112 MB)
│   ├── checkpoint_epoch_*.pt
│   └── validation_results/
│       ├── test_metrics.json              # GLORYS test metrics
│       ├── argo_validation_metrics.json   # Argo independent metrics
│       └── test_predictions.nc
├── docs/
│   ├── research-summary.md
│   ├── architecture-decisions.md          # Updated with actual decisions
│   ├── decisions.md                       # 13-entry decision log
│   └── data-sources.md
├── experiments.md                         # Full experiment log
├── train_full.py                          # 100-epoch training (resumes)
├── validate_test.py                       # GLORYS test validation
├── validate_argo.py                       # Argo independent validation
├── run.py                                 # CLI: preprocess/train/validate/demo
├── STATUS.md                              # THIS FILE
├── MODEL_REPORT.md                        # Full model report
└── PRD.md / architecture.md / design.md / phase.md / rules.md
```

---

## 7. Team & Competition Info

- **Competition:** Smart India Hackathon 2026
- **Problem ID:** SIH26066
- **Team:** Neural Shadows
- **Project:** OceanEmbed — 3D Ocean Temperature Reconstruction
- **Domain:** North Indian Ocean (0–25°N, 45–100°E)
- **Key constraint:** Demo must work fully offline during judging
- **GPU:** NVIDIA RTX 4050 Laptop (6 GB VRAM, CUDA 12.1)
