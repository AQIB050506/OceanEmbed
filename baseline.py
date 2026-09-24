#!/usr/bin/env python3
"""
Baseline models for OceanEmbed (Phase 2 requirement).

Two baselines:
  1. Climatology — predict the training-set mean temperature profile everywhere.
  2. Linear regression — per-depth OLS mapping from surface channels
     (sst, ssh, sss, wind_u, wind_v) to temperature at each depth level.

Evaluated on:
  - GLORYS held-out test set (same metrics as validate_test.py)
  - Raw INCOIS ARGO float observations (same matching as validate_argo.py)

Every later model must beat these numbers.
"""
import sys
import json
import time
import logging
import numpy as np
import torch
import xarray as xr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PROCESSED_DIR, MODELS_DIR, DEPTH_LEVELS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION,
)
from src.preprocessing.dataset import OceanTemperatureDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

TEMPORAL_WINDOW = 5
RAW_ARGO_PATH = Path(
    "D:/OceanEmbed/data/incois_argo/Indian_ARGO_Floats_4298_d1fb_dfa5_U1789797672771.nc"
)
OUTPUT_DIR = MODELS_DIR / "validation_results"
DEPTH_TOL = 10.0
TRAIN_SUBSAMPLE = 4  # use every Nth training day for fitting (speed/memory)


def load_split(surface_name: str, target_name: str) -> OceanTemperatureDataset:
    ds = OceanTemperatureDataset(
        surface_path=PROCESSED_DIR / surface_name,
        target_path=PROCESSED_DIR / target_name,
        temporal_window=TEMPORAL_WINDOW,
        augment=False,
        normalize=True,
    )
    log.info("Loaded %s: %d samples", surface_name, len(ds))
    return ds


def get_surface_features(ds: OceanTemperatureDataset, idx: int) -> np.ndarray:
    """Return (H, W, 5) physical-unit surface features for sample idx (center day)."""
    # inputs: (C, T, H, W) normalized; take center temporal slice
    inputs, _ = ds[idx]
    arr = inputs.numpy()  # (5, T, H, W)
    center = arr.shape[1] // 2
    feat_norm = arr[:, center, :, :]  # (5, H, W)
    feat = np.empty_like(feat_norm)
    for c, var in enumerate(["sst", "ssh", "sss", "wind_u", "wind_v"]):
        mean = ds.stats[var]["mean"]
        std = ds.stats[var]["std"]
        feat[c] = feat_norm[c] * std + mean
    return np.transpose(feat, (1, 2, 0))  # (H, W, 5)


def get_target_physical(ds: OceanTemperatureDataset, idx: int) -> np.ndarray:
    """Return (D, H, W) physical-unit temperature target for sample idx."""
    _, target = ds[idx]
    t = target.numpy()
    mean = ds.stats["thetao"]["mean"]
    std = ds.stats["thetao"]["std"]
    return t * std + mean


def fit_baselines(train_ds: OceanTemperatureDataset):
    """Fit climatology profile and linear regression on a subsample of train set."""
    n = len(train_ds)
    indices = list(range(0, n, TRAIN_SUBSAMPLE))
    log.info(
        "Fitting baselines on %d/%d training days (subsample=%d)...",
        len(indices), n, TRAIN_SUBSAMPLE,
    )

    # Accumulators for least squares: X^T X and X^T Y
    # features: 5 (+1 bias) -> 6; targets: 15 depths
    n_feat = 6  # [sst, ssh, sss, wind_u, wind_v, 1]
    XtX = np.zeros((n_feat, n_feat), dtype=np.float64)
    XTY = np.zeros((n_feat, len(DEPTH_LEVELS)), dtype=np.float64)

    # Climatology accumulators
    profile_sum = np.zeros(len(DEPTH_LEVELS), dtype=np.float64)
    profile_count = 0

    H = train_ds.n_lat
    W = train_ds.n_lon

    for k, idx in enumerate(indices):
        feat = get_surface_features(train_ds, idx)  # (H, W, 5)
        target = get_target_physical(train_ds, idx)  # (D, H, W)

        # Flatten spatial dims
        X = feat.reshape(-1, 5).astype(np.float64)  # (H*W, 5)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # (H*W, 6)
        Y = target.reshape(len(DEPTH_LEVELS), -1).T  # (H*W, D)

        XtX += Xb.T @ Xb
        XTY += Xb.T @ Y

        # Climatology: mean profile over this day's valid cells
        valid = ~np.isnan(target).any(axis=0)
        if valid.any():
            profile_sum += target[:, valid].mean(axis=1)
            profile_count += 1

        if (k + 1) % 50 == 0:
            log.info("  Fitted on %d/%d days", k + 1, len(indices))

    # Solve normal equations
    beta = np.linalg.solve(XtX + 1e-6 * np.eye(n_feat), XTY)  # (6, D)
    climatology = profile_sum / max(profile_count, 1)  # (D,)

    log.info("Linear regression fit complete. Beta shape: %s", beta.shape)
    log.info("Climatology profile: %s", np.round(climatology, 2))
    return beta, climatology


def predict_linear(feat: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """feat: (H, W, 5) -> pred: (D, H, W)"""
    H, W, _ = feat.shape
    X = feat.reshape(-1, 5).astype(np.float64)
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    Y = Xb @ beta  # (H*W, D)
    return Y.T.reshape(len(DEPTH_LEVELS), H, W)


def predict_climatology(climatology: np.ndarray, shape_hw: tuple) -> np.ndarray:
    H, W = shape_hw
    return np.broadcast_to(climatology[:, None, None], (len(DEPTH_LEVELS), H, W)).copy()


def run_predictions(ds: OceanTemperatureDataset, beta, climatology):
    """Generate baseline predictions for every sample in ds."""
    preds_lin = []
    preds_clim = []
    targets = []
    dates = []

    for i in range(len(ds)):
        feat = get_surface_features(ds, i)
        target = get_target_physical(ds, i)
        preds_lin.append(predict_linear(feat, beta))
        preds_clim.append(predict_climatology(climatology, feat.shape[:2]))
        targets.append(target)
        dates.append(str(ds.get_date(i))[:10])
        if (i + 1) % 50 == 0:
            log.info("  Predicted %d/%d samples", i + 1, len(ds))

    return (
        np.stack(preds_lin),
        np.stack(preds_clim),
        np.stack(targets),
        dates,
    )


def compute_glorys_metrics(preds: np.ndarray, targets: np.ndarray, label: str) -> dict:
    """Same metrics as validate_test.py compute_metrics."""
    n_depth = preds.shape[1]
    results = {}
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
        "model": label,
    }
    return results


def load_argo_raw():
    log.info("Loading raw ARGO from %s", RAW_ARGO_PATH.name)
    ds = xr.open_dataset(RAW_ARGO_PATH)
    temp_qc = ds["TEMP_QC"].values.astype(str)
    pres_qc = ds["PRES_QC"].values.astype(str)
    good_mask = (
        ((temp_qc == "1") | (temp_qc == "2") | (temp_qc == " "))
        & ((pres_qc == "1") | (pres_qc == "2") | (pres_qc == " "))
    )
    mask = (
        good_mask
        & (ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX)
        & (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX)
        & (ds.PRES >= 0) & (ds.PRES <= 500)
        & (~np.isnan(ds.TEMP.values))
    )
    times = ds.time.values[mask]
    lats = ds.latitude.values[mask]
    lons = ds.longitude.values[mask]
    pres = ds.PRES.values[mask]
    temps = ds.TEMP.values[mask]
    ds.close()
    depths = pres / 1.02
    dates = np.array([str(t)[:10] for t in times])
    log.info("Filtered ARGO: %d obs", len(dates))
    return dates, lats, lons, depths, temps


def evaluate_argo(preds, test_dates, argo_dates, argo_lats, argo_lons, argo_depths, argo_temps):
    """Same point-matching logic as validate_argo.py evaluate()."""
    lats = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
    lons = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
    depth_levels = np.array(DEPTH_LEVELS)

    test_date_set = set(test_dates)
    argo_date_set = set(argo_dates)
    common_dates = sorted(test_date_set & argo_date_set)
    log.info("Common dates: %d", len(common_dates))

    argo_date_idx = {}
    for i, d in enumerate(argo_dates):
        if d in argo_date_set:
            argo_date_idx.setdefault(d, []).append(i)

    depth_metrics = {d: {"rmse": [], "mae": [], "bias": [], "corr": []} for d in DEPTH_LEVELS}
    total = 0

    for date_str in common_dates:
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
            (lat_idx >= 0) & (lat_idx < len(lats))
            & (lon_idx >= 0) & (lon_idx < len(lons))
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
            total += 1

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
        "n_total_points": total,
        "n_common_dates": len(common_dates),
    }
    return summary


def print_table(glorys_metrics, argo_metrics, label):
    log.info("")
    log.info("=" * 78)
    log.info("BASELINE: %s", label.upper())
    log.info("=" * 78)

    log.info("")
    log.info("-- GLORYS Test Set --")
    o = glorys_metrics["overall"]
    log.info("Overall: RMSE=%.4f  MAE=%.4f  R2=%.4f", o["rmse"], o["mae"], o["r2"])
    log.info("%-8s  %-8s  %-8s  %-8s  %-8s" % ("Depth(m)", "RMSE", "MAE", "R2", "Corr"))
    log.info("-" * 50)
    for d in DEPTH_LEVELS:
        if d in glorys_metrics:
            m = glorys_metrics[d]
            log.info("%-8d  %-8.4f  %-8.4f  %-8.4f  %-8.4f",
                     m["depth_m"], m["rmse"], m["mae"], m["r2"], m["correlation"])

    log.info("")
    log.info("-- ARGO Independent Validation --")
    ao = argo_metrics.get("overall", {})
    log.info("Overall: RMSE=%.4f  MAE=%.4f  Corr=%.4f  (points=%d, dates=%d)",
             ao.get("mean_rmse") or 0, ao.get("mean_mae") or 0,
             ao.get("mean_correlation") or 0,
             ao.get("n_total_points") or 0, ao.get("n_common_dates") or 0)
    log.info("%-8s  %-8s  %-8s  %-8s  %-8s" % ("Depth(m)", "RMSE", "MAE", "Bias", "Corr"))
    log.info("-" * 50)
    for d in DEPTH_LEVELS:
        if d in argo_metrics and isinstance(argo_metrics[d], dict) and "rmse" in argo_metrics[d]:
            m = argo_metrics[d]
            log.info("%-8d  %-8.4f  %-8.4f  %-8.4f  %-8.4f",
                     d, m["rmse"], m["mae"], m["bias"], m["correlation"])
    log.info("=" * 78)


def main():
    t_start = time.time()

    log.info("Loading training set for baseline fitting...")
    train_ds = load_split("train_surface.nc", "train_target.nc")
    beta, climatology = fit_baselines(train_ds)

    log.info("Loading test set...")
    test_ds = load_split("test_surface.nc", "test_target.nc")

    log.info("Generating baseline predictions on test set (%d samples)...", len(test_ds))
    t0 = time.time()
    preds_lin, preds_clim, targets, test_dates = run_predictions(test_ds, beta, climatology)
    log.info("Prediction time: %.1fs", time.time() - t0)

    # GLORYS metrics
    glorys_lin = compute_glorys_metrics(preds_lin, targets, "linear_regression")
    glorys_clim = compute_glorys_metrics(preds_clim, targets, "climatology")

    # ARGO metrics
    argo_dates, argo_lats, argo_lons, argo_depths, argo_temps = load_argo_raw()
    argo_lin = evaluate_argo(preds_lin, test_dates, argo_dates, argo_lats, argo_lons, argo_depths, argo_temps)
    argo_clim = evaluate_argo(preds_clim, test_dates, argo_dates, argo_lats, argo_lons, argo_depths, argo_temps)

    print_table(glorys_lin, argo_lin, "linear_regression")
    print_table(glorys_clim, argo_clim, "climatology")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "linear_regression": {
            "description": "Per-depth OLS: [sst, ssh, sss, wind_u, wind_v, 1] -> T(z)",
            "train_subsample": TRAIN_SUBSAMPLE,
            "glorys_test": glorys_lin,
            "argo": argo_lin,
        },
        "climatology": {
            "description": "Training-set mean temperature profile (constant in space/time)",
            "glorys_test": glorys_clim,
            "argo": argo_clim,
        },
        "dl_model_reference": {
            "description": "AttentionUNet3D (trained 100 epochs) for comparison",
            "glorys_test_overall": {"rmse": 0.083, "mae": 0.045, "r2": 0.989},
            "argo_overall": {"mean_rmse": 1.02, "mean_correlation": 0.83},
        },
    }
    out_path = OUTPUT_DIR / "baseline_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Saved baseline metrics to %s", out_path)

    # Also save fitted parameters for reuse
    np.savez(
        OUTPUT_DIR / "baseline_params.npz",
        beta=beta,
        climatology=climatology,
        depth_levels=np.array(DEPTH_LEVELS),
    )
    log.info("Saved fitted parameters to %s", OUTPUT_DIR / "baseline_params.npz")

    log.info("Total time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
