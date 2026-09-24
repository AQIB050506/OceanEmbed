#!/usr/bin/env python3
"""
Validation against GLORYS test set (2024-2025).
No INCOIS Argo needed — compares model predictions to held-out GLORYS targets.
"""
import sys
import json
import time
import logging
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from src.config import (
    PROCESSED_DIR, MODELS_DIR, DEPTH_LEVELS, MODEL_CONFIG,
    LAT_MIN, LON_MIN, RESOLUTION,
)
from src.reconstruction.model import AttentionUNet3D, count_parameters
from src.preprocessing.dataset import OceanTemperatureDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEVICE = "cuda"
BATCH_SIZE = 2
TEMPORAL_WINDOW = 5
OUTPUT_DIR = MODELS_DIR / "validation_results"


def load_model():
    ckpt = torch.load(MODELS_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    model = AttentionUNet3D().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info("Model loaded from epoch %d (val_loss=%.6f)",
             ckpt.get("epoch", -1), ckpt.get("best_val_loss", -1))
    return model


def load_test_data():
    ds = OceanTemperatureDataset(
        surface_path=PROCESSED_DIR / "test_surface.nc",
        target_path=PROCESSED_DIR / "test_target.nc",
        temporal_window=TEMPORAL_WINDOW,
        augment=False,
        normalize=True,
    )
    log.info("Test set: %d samples, %d x %d spatial", len(ds), ds.n_lat, ds.n_lon)
    return ds


@torch.no_grad()
def run_inference(model, dataset):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )
    all_preds = []
    all_targets = []
    all_dates = []

    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(DEVICE, non_blocking=True)
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())

        for j in range(inputs.shape[0]):
            idx = i * BATCH_SIZE + j
            if idx < len(dataset):
                all_dates.append(dataset.get_date(idx))

        if (i + 1) % 100 == 0:
            log.info("  Inference: %d/%d batches", i + 1, len(loader))

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    log.info("Inference complete: %d predictions, shape %s", preds.shape[0], preds.shape)
    return preds, targets, all_dates


def compute_metrics(preds, targets):
    n_depth = preds.shape[1]
    results = {}

    # Per-depth metrics
    for d in range(n_depth):
        p = preds[:, d].flatten()
        t = targets[:, d].flatten()
        valid = ~(np.isnan(p) | np.isnan(t))
        if valid.sum() < 10:
            continue
        p, t = p[valid], t[valid]

        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        bias = float(np.mean(p - t))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))
        corr = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 else 0.0

        results[DEPTH_LEVELS[d]] = {
            "depth_m": DEPTH_LEVELS[d],
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "bias": round(bias, 6),
            "r2": round(r2, 6),
            "correlation": round(corr, 6),
            "n_pixels": int(valid.sum()),
        }

    # Overall metrics
    p_all = preds.flatten()
    t_all = targets.flatten()
    valid = ~(np.isnan(p_all) | np.isnan(t_all))
    p_v, t_v = p_all[valid], t_all[valid]
    overall_rmse = float(np.sqrt(np.mean((p_v - t_v) ** 2)))
    overall_mae = float(np.mean(np.abs(p_v - t_v)))
    ss_res = np.sum((t_v - p_v) ** 2)
    ss_tot = np.sum((t_v - np.mean(t_v)) ** 2)
    overall_r2 = float(1 - ss_res / (ss_tot + 1e-8))

    results["overall"] = {
        "rmse": round(overall_rmse, 6),
        "mae": round(overall_mae, 6),
        "r2": round(overall_r2, 6),
        "n_pixels": int(valid.sum()),
    }

    return results


def save_results(metrics, preds, targets, dates):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Metrics saved to %s", OUTPUT_DIR / "test_metrics.json")

    # Save predictions as NetCDF
    import xarray as xr
    lats = np.arange(LAT_MIN, LAT_MIN + RESOLUTION * preds.shape[2], RESOLUTION)
    lons = np.arange(LON_MIN, LON_MIN + RESOLUTION * preds.shape[3], RESOLUTION)
    ds = xr.Dataset(
        {
            "temperature_pred": (["time", "depth", "latitude", "longitude"], preds.astype(np.float32)),
            "temperature_target": (["time", "depth", "latitude", "longitude"], targets.astype(np.float32)),
        },
        coords={
            "time": dates[:preds.shape[0]],
            "depth": DEPTH_LEVELS,
            "latitude": lats,
            "longitude": lons,
        },
    )
    ds.to_netcdf(OUTPUT_DIR / "test_predictions.nc")
    log.info("Predictions saved to %s", OUTPUT_DIR / "test_predictions.nc")


def print_summary(metrics):
    log.info("=" * 60)
    log.info("VALIDATION RESULTS — GLORYS Test Set (2024-2025)")
    log.info("=" * 60)

    log.info("")
    log.info("Overall:  RMSE=%.4f  MAE=%.4f  R²=%.4f",
             metrics["overall"]["rmse"], metrics["overall"]["mae"], metrics["overall"]["r2"])
    log.info("")
    log.info("%-8s  %-8s  %-8s  %-8s  %-8s  %-8s" % ("Depth(m)", "RMSE", "MAE", "Bias", "R²", "Corr"))
    log.info("-" * 56)

    for d in DEPTH_LEVELS:
        if d in metrics:
            m = metrics[d]
            log.info("%-8d  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8.4f",
                     m["depth_m"], m["rmse"], m["mae"], m["bias"], m["r2"], m["correlation"])

    log.info("=" * 60)


def main():
    log.info("Device: %s", torch.cuda.get_device_name(0))

    model = load_model()
    dataset = load_test_data()

    t0 = time.time()
    preds, targets, dates = run_inference(model, dataset)
    elapsed = time.time() - t0
    log.info("Inference time: %.1fs (%.2f ms/sample)", elapsed, elapsed / len(preds) * 1000)

    with open(PROCESSED_DIR / "normalization_stats.json") as f:
        norm_stats = json.load(f)
    thetao_mean = norm_stats["thetao"]["mean"]
    thetao_std = norm_stats["thetao"]["std"]
    log.info("Denormalization: mean=%.4f, std=%.4f", thetao_mean, thetao_std)

    preds = preds * thetao_std + thetao_mean
    targets = targets * thetao_std + thetao_mean
    log.info("Predictions and targets denormalized to physical units (°C)")

    metrics = compute_metrics(preds, targets)
    save_results(metrics, preds, targets, dates)
    print_summary(metrics)


if __name__ == "__main__":
    main()
