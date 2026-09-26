<div align="center">
  <h1>OceanEmbed</h1>
  <p>
    <strong>3D subsurface ocean temperature reconstruction from surface satellite observations</strong><br>
    North Indian Ocean · 0–450 m · 15 depth levels · 0.25° grid · daily
  </p>
  <p>
    <a href="https://oceanembed-pied.vercel.app">
      <img src="https://img.shields.io/badge/Launch-Live%20Demo%20%E2%86%92-0f172a?style=for-the-badge&logo=vercel&logoColor=white" alt="Launch the live demo">
    </a>
  </p>
  <p><code>https://oceanembed-pied.vercel.app</code></p>
</div>

<p align="center">
  <a href="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white"><img src="https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://img.shields.io/badge/ONNX%20Runtime-CPU-005CED?logo=onnx&logoColor=white"><img src="https://img.shields.io/badge/ONNX%20Runtime-CPU-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime"></a>
  <a href="https://img.shields.io/badge/tests-5%20passing-brightgreen"><img src="https://img.shields.io/badge/tests-5%20passing-brightgreen" alt="tests"></a>
  <a href="https://oceanembed-pied.vercel.app"><img src="https://img.shields.io/badge/Live%20Demo-oceanembed--pied.vercel.app-111827?logo=vercel" alt="Live demo"></a>
  <a href="https://img.shields.io/badge/Smart%20India%20Hackathon-2026-e11d48"><img src="https://img.shields.io/badge/Smart%20India%20Hackathon-2026-e11d48" alt="SIH 2026"></a>
</p>

<p align="center">
  🌊 <strong><a href="https://oceanembed-pied.vercel.app">Live Demo</a></strong> ·
  📊 <a href="MODEL_REPORT.md">Model Report</a> ·
  🛠 <a href="SETUP.md">Setup Guide</a> ·
  📁 <a href="docs/">Documentation</a>
</p>

---

OceanEmbed reconstructs the full vertical temperature structure of the ocean from **surface fields only** (SST, SSH, SSS, surface winds) — turning cheap, always-available satellite data into the subsurface view that until now required sparse, expensive Argo floats.

## Highlights

- **98.9% variance explained** (R²) on the held-out GLORYS test set; **RMSE 0.083 °C** overall
- **Independently validated against Argo floats**: mean RMSE 1.02 °C across 56,925 colocations (345 dates)
- **9.8M-parameter** attention-enhanced 3D U-Net — trains on a 6 GB laptop GPU, infers on CPU in **0.6 s**
- **Operational-ready demo**: live satellite fetches, 14-day time playback, Argo overlay, arbitrary transects
- **Ships deployable**: bundled ONNX model + precomputed assets — clone and run, no dataset required

## Results

Held-out test set (GLORYS reanalysis, 2024–2025, 346 samples):

| Zone | Depths | Avg R² | Avg RMSE |
|------|--------|--------|----------|
| Surface | 0–30 m | 0.991 | 0.052 °C |
| Upper thermocline | 50–75 m | 0.924 | 0.107 °C |
| Main thermocline | 100–150 m | 0.848 | 0.124 °C |
| Deep | 200–450 m | 0.986 | 0.059 °C |
| **Overall** | **0–450 m** | **0.989** | **0.083 °C** |

Independent Argo-float validation (2024–2025): mean correlation **0.83**, mean RMSE **1.02 °C** at all 15 levels.

Full per-depth tables: [MODEL_REPORT.md](MODEL_REPORT.md) · raw metrics: [`models/validation_results/`](models/validation_results/)

## How It Works

```
Satellite surface fields          AttentionUNet3D              Application products
┌──────────────────────┐        ┌────────────────────┐        ┌─────────────────────┐
│ SST · SSH · SSS      │        │ 3× 3D encoder      │        │ Temperature profiles│
│ Wind_U · Wind_V      │ ─────▶ │ CBAM attention     │ ─────▶ │ Ocean heat content  │
│ (B, 5, 5, 101, 221)  │        │ 3× 3D decoder      │        │ Marine heatwaves    │
│ 5 days × spatial     │        │ depth projection   │        │ Cyclone risk class  │
└──────────────────────┘        └────────────────────┘        └─────────────────────┘
                                                                  (15, 101, 221)
```

| Spec | Value |
|------|-------|
| Domain | 0–25°N, 45–100°E at 0.25° (101 × 221 grid) |
| Depth levels | 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 450 m |
| Parameters | 9,791,217 (≈ 9.8 M) |
| Training | AdamW + cosine schedule, depth-weighted & uncertainty-aware loss, 100 epochs |
| Inference | ONNX Runtime CPU ≈ 0.56 s/sample (local) · PyTorch CUDA ≈ 60 ms/sample |

## Interactive Demo

The production demo runs on [Vercel](https://oceanembed-pied.vercel.app) — FastAPI backend, serverless Python, static frontend:

- **3D + map views** — Three.js ocean volume with a Leaflet minimap (vendor-bundled, works offline)
- **Live nowcast** — daily SST & winds fetched from Open-Meteo, inference on every request
- **14-day time playback** — scrub or play historical reconstructions
- **Argo overlay** — 500 recent in-domain float profiles; click one to snap the cursor and see model-vs-float RMSE
- **Transect explorer** — click any two points for a 15-depth section along the path (distance, land fraction)
- **Decision products** — OHC, marine-heatwave detection, cyclone-intensification risk at any point

Deep-link examples: [`?argo=1`](https://oceanembed-pied.vercel.app/?argo=1) · [`?tr=5,50,15,90`](https://oceanembed-pied.vercel.app/?tr=5,50,15,90) · `?date=YYYY-MM-DD` (any date in the last 14 days)

## Quick Start

### Run the demo (CPU-only, no dataset needed)

The repository bundles `demo/online/assets/` (ONNX model, land mask, normalization stats, Argo index) — **no GPU, no training data required**.

```bash
git clone <repo-url>
cd OceanEmbed

python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

pip install fastapi uvicorn numpy scipy requests onnxruntime netCDF4

python demo/online/start.py
# → http://localhost:8000
```

> The engine automatically prefers `assets/model.onnx` (CPU). If you have a local `models/best_model.pt`, deleting the ONNX file falls back to PyTorch/CUDA.

### Full development environment (training, preprocessing, tests)

```bash
pip install -r requirements.txt
pytest tests -q          # 5 tests
```

For CUDA/PyTorch setup, dataset download and training, see **[SETUP.md](SETUP.md)** and [`train_full.py`](train_full.py).

### Deploy to Vercel

The project is pre-configured (`pyproject.toml` → `[tool.vercel]`, `.vercelignore`):

```bash
npx vercel              # preview deployment
npx vercel --prod       # production
```

## Project Structure

```
├── src/
│   ├── config.py                 # Grid, paths, hyperparameters
│   ├── reconstruction/
│   │   ├── model.py              # AttentionUNet3D (9.8M params)
│   │   └── train.py              # Losses, trainer, checkpointing
│   ├── preprocessing/            # Regridding, harmonization, datasets
│   ├── validation/               # Evaluation & Argo validation
│   └── application/
│       └── products.py           # OHC, marine heatwave, cyclone risk
├── demo/
│   ├── online/                   # ★ Production demo (Vercel)
│   │   ├── server.py             #   FastAPI app — all /api/* endpoints
│   │   ├── realtime_fetcher.py   #   Open-Meteo live & archive fetches
│   │   ├── frontend/             #   Single-page dashboard (vendored libs)
│   │   ├── assets/               #   ONNX model + land mask + stats + Argo CSV
│   │   └── start.py              #   Local launcher (:8000)
│   ├── frontend/                 # Offline demo UI
│   └── backend/                  # Offline demo API
├── tests/                        # Model smoke tests (pytest)
├── docs/                         # Architecture, decisions, research
├── models/validation_results/    # Per-depth metrics (JSON)
├── export_onnx.py                # best_model.pt → assets/model.onnx
├── precompute_assets.py          # Regenerate deploy assets
├── verify_onnx_parity.py         # torch ↔ ONNX numerical gate
├── pyproject.toml                # Vercel runtime config
└── requirements.txt              # Full local/dev dependencies
```

## Data & Model Weights

| Artifact | In repo? | Notes |
|----------|----------|-------|
| `demo/online/assets/model.onnx` (39 MB) | ✅ | Inference model for demo & Vercel |
| Precomputed assets (mask, stats, Argo CSV) | ✅ | Regenerate with `precompute_assets.py` |
| `models/best_model.pt` (112 MB) | ❌ | Exceeds GitHub limit — retrain via `train_full.py` or request from the team |
| `data/` (GLORYS, satellite products, ≈ 277 GB) | ❌ | Download with `download_*.py` (Copernicus credentials required) |

## Documentation

| Document | Contents |
|----------|----------|
| [MODEL_REPORT.md](MODEL_REPORT.md) | Problem, architecture, training, full metrics |
| [SETUP.md](SETUP.md) | Environment setup, CUDA, dataset download |
| [PRD.md](PRD.md) · [architecture.md](architecture.md) · [design.md](design.md) | Requirements & design |
| [docs/decisions.md](docs/decisions.md) | Architecture decision records |
| [docs/data-sources.md](docs/data-sources.md) | Dataset catalogue |
| [STATUS.md](STATUS.md) · [experiments.md](experiments.md) | Progress log & experiment history |

## Team

**Neural Shadows** — Smart India Hackathon 2026 · Problem ID **SIH26066**
Organization: Ministry of Earth Sciences / INCOIS

| | |
|---|---|
| Frameworks | FastAPI · PyTorch · ONNX Runtime · Three.js · Leaflet · Chart.js |
| Infra | Vercel serverless (Python 3.12) · Open-Meteo & Argo GDAC (keyless APIs) |
| Hardware (training) | NVIDIA RTX 4050 (6 GB) · PyTorch 2.5.1+cu121 |
