#!/usr/bin/env python3
"""
INCOIS Argo Validation - compares model predictions directly against
raw Argo float observations (bypasses the broken gridded file).

For each test date, finds Argo observations on that date within the domain,
and compares the model's predicted temperature at that grid cell.
"""
import sys
import json
import time
import logging
import numpy as np
import torch
import xarray as xr
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PROCESSED_DIR, MODELS_DIR, DEPTH_LEVELS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION,
)
from src.reconstruction.model import AttentionUNet3D
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
RAW_ARGO_PATH = Path("D:/OceanEmbed/data/incois_argo/Indian_ARGO_Floats_4298_d1fb_dfa5_U1789797672771.nc")
OUTPUT_DIR = MODELS_DIR / "validation_results"
DEPTH_TOL = 10.0


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
    log.info("Test set: %d samples", len(ds))
    return ds


def load_argo_raw():
    log.info("Loading raw Argo from %s (%.1f MB)",
             RAW_ARGO_PATH, RAW_ARGO_PATH.stat().st_size / 1e6)
    ds = xr.open_dataset(RAW_ARGO_PATH)

    temp_qc = ds["TEMP_QC"].values.astype(str)
    pres_qc = ds["PRES_QC"].values.astype(str)
    good_mask = (
        ((temp_qc == "1") | (temp_qc == "2") | (temp_qc == " ")) &
        ((pres_qc == "1") | (pres_qc == "2") | (pres_qc == " "))
    )
    mask = (
        good_mask &
        (ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX) &
        (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX) &
        (ds.PRES >= 0) & (ds.PRES <= 500) &
        (~np.isnan(ds.TEMP.values))
    )

    times = ds.time.values[mask]
    lats = ds.latitude.values[mask]
    lons = ds.longitude.values[mask]
    pres = ds.PRES.values[mask]
    temps = ds.TEMP.values[mask]
    ds.close()

    depths = pres / 1.02
    dates = np.array([str(t)[:10] for t in times])

    log.info("Filtered Argo: %d observations, dates %s to %s",
             len(dates), sorted(dates)[0], sorted(dates)[-1])
    return dates, lats, lons, depths, temps


@torch.no_grad()
def run_inference(model, dataset):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )
    all_preds = []
    all_dates = []

    for i, (inputs, _) in enumerate(loader):
        inputs = inputs.to(DEVICE, non_blocking=True)
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())

        for j in range(inputs.shape[0]):
            idx = i * BATCH_SIZE + j
            if idx < len(dataset):
                all_dates.append(str(dataset.get_date(idx))[:10])

        if (i + 1) % 100 == 0:
            log.info("  Inference: %d/%d batches", i + 1, len(loader))

    preds = np.concatenate(all_preds, axis=0)
    log.info("Inference complete: %d predictions, shape %s", preds.shape[0], preds.shape)
    return preds, all_dates


def evaluate(preds, test_dates, argo_dates, argo_lats, argo_lons, argo_depths, argo_temps):
    lats = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
    lons = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
    depth_levels = np.array(DEPTH_LEVELS)

    test_date_set = set(test_dates)
    argo_date_set = set(argo_dates)
    common_dates = sorted(test_date_set & argo_date_set)
    log.info("Common dates between test set and Argo: %d", len(common_dates))

    log.info("Building argo date index...")
    argo_date_idx = {}
    for i, d in enumerate(argo_dates):
        if d in argo_date_set:
            if d not in argo_date_idx:
                argo_date_idx[d] = []
            argo_date_idx[d].append(i)
    log.info("Argo date index built: %d dates indexed", len(argo_date_idx))

    depth_levels = np.array(DEPTH_LEVELS)
    depth_metrics = {d: {"rmse": [], "mae": [], "bias": [], "corr": []} for d in DEPTH_LEVELS}
    total_comparisons = 0

    for ci, date_str in enumerate(common_dates):
        indices = argo_date_idx[date_str]
        d_lats = argo_lats[indices]
        d_lons = argo_lons[indices]
        d_depths = argo_depths[indices]
        d_temps = argo_temps[indices]

        test_idx = test_dates.index(date_str)
        pred_grid = preds[test_idx]

        lat_idx = np.round((d_lats - LAT_MIN) / RESOLUTION).astype(int)
        lon_idx = np.round((d_lons - LON_MIN) / RESOLUTION).astype(int)

        valid_obs = (
            (lat_idx >= 0) & (lat_idx < len(lats)) &
            (lon_idx >= 0) & (lon_idx < len(lons))
        )

        for obs_i in np.where(valid_obs)[0]:
            li, lo = lat_idx[obs_i], lon_idx[obs_i]
            obs_depth = d_depths[obs_i]
            obs_temp = d_temps[obs_i]

            depth_idx = np.argmin(np.abs(depth_levels - obs_depth))
            if abs(depth_levels[depth_idx] - obs_depth) > DEPTH_TOL:
                continue

            pred_temp = pred_grid[depth_idx, li, lo]
            if np.isnan(pred_temp):
                continue

            depth = DEPTH_LEVELS[depth_idx]
            depth_metrics[depth]["rmse"].append((pred_temp - obs_temp) ** 2)
            depth_metrics[depth]["mae"].append(abs(pred_temp - obs_temp))
            depth_metrics[depth]["bias"].append(pred_temp - obs_temp)
            depth_metrics[depth]["corr"].append((pred_temp, obs_temp))
            total_comparisons += 1

        if (ci + 1) % 20 == 0:
            log.info("  Compared %d/%d dates, %d total point comparisons",
                     ci + 1, len(common_dates), total_comparisons)

    log.info("Total point comparisons: %d", total_comparisons)

    summary = {}
    for depth in DEPTH_LEVELS:
        m = depth_metrics[depth]
        n = len(m["rmse"])
        if n < 5:
            continue
        rmse = float(np.sqrt(np.mean(m["rmse"])))
        mae = float(np.mean(m["mae"]))
        bias = float(np.mean(m["bias"]))
        pairs = np.array(m["corr"])
        corr = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) if len(pairs) > 1 else 0.0

        summary[depth] = {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "bias": round(bias, 6),
            "correlation": round(corr, 6),
            "n_points": n,
        }

    all_rmse = [v["rmse"] for v in summary.values()]
    all_corr = [v["correlation"] for v in summary.values()]
    all_mae = [v["mae"] for v in summary.values()]

    summary["overall"] = {
        "mean_rmse": round(float(np.mean(all_rmse)), 6) if all_rmse else None,
        "mean_correlation": round(float(np.mean(all_corr)), 6) if all_corr else None,
        "mean_mae": round(float(np.mean(all_mae)), 6) if all_mae else None,
        "n_depths": len(all_rmse),
        "n_total_points": total_comparisons,
        "n_common_dates": len(common_dates),
    }

    return summary


def print_summary(metrics):
    log.info("")
    log.info("=" * 70)
    log.info("INCOIS ARGO VALIDATION RESULTS")
    log.info("=" * 70)
    log.info("")
    log.info("%-8s  %-10s  %-10s  %-10s  %-10s  %-8s" % (
        "Depth(m)", "RMSE", "MAE", "Bias", "Corr", "N"))
    log.info("-" * 62)

    for d in DEPTH_LEVELS:
        if d in metrics and isinstance(metrics[d], dict) and "rmse" in metrics[d]:
            m = metrics[d]
            log.info("%-8d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-8d" % (
                d, m["rmse"], m["mae"], m["bias"], m["correlation"], m["n_points"]))

    log.info("-" * 62)
    o = metrics.get("overall", {})
    log.info("Overall: RMSE=%.4f  MAE=%.4f  Corr=%.4f  (depths=%d, points=%d, dates=%d)" % (
        o.get("mean_rmse") or 0, o.get("mean_mae") or 0,
        o.get("mean_correlation") or 0,
        o.get("n_depths") or 0,
        o.get("n_total_points") or 0,
        o.get("n_common_dates") or 0))
    log.info("=" * 70)


def main():
    log.info("Device: %s", torch.cuda.get_device_name(0))

    model = load_model()
    dataset = load_test_data()

    t0 = time.time()
    preds, test_dates = run_inference(model, dataset)
    log.info("Inference time: %.1fs", time.time() - t0)

    # Denormalize predictions: pred_real = pred_norm * std + mean
    with open(PROCESSED_DIR / "normalization_stats.json") as f:
        norm_stats = json.load(f)
    thetao_mean = norm_stats["thetao"]["mean"]
    thetao_std = norm_stats["thetao"]["std"]
    log.info("Denormalization: mean=%.4f, std=%.4f", thetao_mean, thetao_std)
    preds = preds * thetao_std + thetao_mean
    log.info("Predictions denormalized to physical units (Celsius)")

    argo_dates, argo_lats, argo_lons, argo_depths, argo_temps = load_argo_raw()

    t1 = time.time()
    metrics = evaluate(preds, test_dates, argo_dates, argo_lats, argo_lons, argo_depths, argo_temps)
    log.info("Argo comparison time: %.1fs", time.time() - t1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "argo_validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Metrics saved to %s", OUTPUT_DIR / "argo_validation_metrics.json")

    print_summary(metrics)


if __name__ == "__main__":
    main()
