# OceanEmbed — Setup Guide

Step-by-step instructions to set up the project on a fresh machine.

---

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10+ | `python --version` |
| NVIDIA GPU | 6 GB+ VRAM recommended | `nvidia-smi` |
| CUDA Toolkit | 12.1+ | `nvcc --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |

> **Important:** This project is **GPU-only**. All code runs on CUDA. No CPU fallback.

---

## Step 1: Clone the Repository

```bash
git clone <repo-url>
cd OceanEmbed
```

---

## Step 2: Install PyTorch (CUDA)

Install PyTorch with CUDA 12.1 support FIRST — this is the most important step.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is working:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Step 3: Install Core Dependencies

```bash
pip install numpy xarray netCDF4 scipy pandas h5py
```

---

## Step 4: Install ML & Visualization

```bash
pip install scikit-learn matplotlib seaborn plotly tqdm
```

---

## Step 5: Install Demo Framework

```bash
pip install fastapi uvicorn pydantic python-dotenv
```

---

## Step 6: Install Geospatial Tools (for preprocessing)

```bash
pip install xesmf cartopy
```

> If `xesmf` fails to install via pip, use conda:
> ```bash
> conda install -c conda-forge xesmf cartopy
> ```

---

## Step 7: Install CMEMS Download Tool

```bash
pip install copernicusmarine
```

Authenticate with your Copernicus Marine credentials:
```bash
copernicusmarine login
```

> Register at: https://marine.copernicus.eu

---

## Step 8: Install Dev Tools (optional)

```bash
pip install pytest ipython jupyter
```

---

## Step 9: Verify Installation

```bash
cd D:\OceanEmbed
pytest tests/test_model.py -v
```

All 5 tests should pass.

---

## One-Line Install (all at once)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121 && pip install numpy xarray netCDF4 scipy pandas h5py scikit-learn matplotlib seaborn plotly tqdm fastapi uvicorn pydantic python-dotenv xesmf cartopy copernicusmarine pytest
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `torch.cuda.is_available()` returns False | Reinstall PyTorch with the cu121 index URL above |
| CUDA out of memory | Reduce `batch_size` in `src/config.py` (currently 2 for 6GB GPU) |
| `xesmf` install fails | Use conda: `conda install -c conda-forge xesmf` |
| `copernicusmarine` auth fails | Run `copernicusmarine login` and enter credentials |
| Tests fail on different GPU | Adjust `embed_dim`, `n_encoder_layers`, `n_decoder_layers` in config.py for your VRAM |
