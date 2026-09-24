from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ─── Domain Bounds (North Indian Ocean — matches downloaded data) ────────────
LAT_MIN = 0.0     # degrees North
LAT_MAX = 25.0
LON_MIN = 45.0    # degrees East
LON_MAX = 100.0
RESOLUTION = 0.25  # degrees

# ─── Depth Levels (15 levels from GLORYS 0–454m range) ──────────────────────
DEPTH_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 450]
N_DEPTH_LEVELS = len(DEPTH_LEVELS)

# ─── Temporal Range ──────────────────────────────────────────────────────────
TRAIN_START = "2010-01-01"
TRAIN_END = "2025-12-31"

# ─── Train/Validation Split ──────────────────────────────────────────────────
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# ─── Actual Data Paths (where downloaded files live) ─────────────────────────
GLORYS_DIR = DATA_DIR / "glorys"
SST_FILE = DATA_DIR / "sst" / "SST_NorthIndianOcean_2010-2025.nc"
SSH_FILE = DATA_DIR / "ssh" / "SLA_NorthIndianOcean_2010-2025.nc"
SSS_FILE = DATA_DIR / "sss" / "SSS_NorthIndianOcean_MY_2010-2025.nc"
WIND_FILE = DATA_DIR / "wind" / "WIND_NorthIndianOcean_2010-2025.nc"
CURRENTS_FILE = DATA_DIR / "currents" / "CUR_NorthIndianOcean_2010-2025.nc"

# ─── Input Variables ─────────────────────────────────────────────────────────
INPUT_VARIABLES = {
    "sst": {
        "file": str(SST_FILE),
        "variable_name": "analysed_sst",
        "rename_to": "sst",
        "native_resolution": 0.05,
        "needs_regrid": True,
        "unit": "kelvin",
    },
    "ssh": {
        "file": str(SSH_FILE),
        "variable_name": "sla",
        "rename_to": "ssh",
        "native_resolution": 0.25,
        "needs_regrid": False,
        "unit": "m",
    },
    "sss": {
        "file": str(SSS_FILE),
        "variable_name": "sos",
        "rename_to": "sss",
        "native_resolution": 0.25,
        "needs_regrid": False,
        "unit": "pss-78",
    },
    "wind_u": {
        "file": str(WIND_FILE),
        "variable_name": "eastward_wind",
        "rename_to": "wind_u",
        "native_resolution": 0.25,
        "needs_regrid": False,
        "unit": "m/s",
    },
    "wind_v": {
        "file": str(WIND_FILE),
        "variable_name": "northward_wind",
        "rename_to": "wind_v",
        "native_resolution": 0.25,
        "needs_regrid": False,
        "unit": "m/s",
    },
}

# ─── Target Variables (from GLORYS) ──────────────────────────────────────────
TARGET_VARIABLES = {
    "temperature": "thetao",
    "salinity": "so",
}

# ─── Model Hyperparameters ──────────────────────────────────────────────────
MODEL_CONFIG = {
    "input_channels": 5,        # SST, SSH, SSS, wind_u, wind_v
    "n_depth_levels": N_DEPTH_LEVELS,
    "embed_dim": 64,
    "n_heads": 4,
    "n_encoder_layers": 3,
    "n_decoder_layers": 3,
    "dropout": 0.1,
    "temporal_window": 5,       # days of input sequence (reduced from 26 for 6GB GPU)
    "learning_rate": 1e-4,
    "batch_size": 2,
    "epochs": 100,
    "weight_decay": 1e-5,
    "scheduler": "cosine",
    "warmup_epochs": 5,
}

# ─── Training Config ─────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "device": "cuda",
    "num_workers": 4,
    "pin_memory": True,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 15,
    "checkpoint_interval": 5,   # save every N epochs
    "log_interval": 50,         # log every N batches
}

# ─── Demo Config ─────────────────────────────────────────────────────────────
DEMO_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "model_checkpoint": MODELS_DIR / "best_model.pt",
    "precomputed_dir": PROCESSED_DIR / "demo_outputs",
}
