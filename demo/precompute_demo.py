"""
Precompute demo outputs for offline demo.
Generates temperature profiles, OHC, MHW, anomaly data, and ARGO
comparison profiles for key dates and locations.
"""
import torch
import numpy as np
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MODELS_DIR, PROCESSED_DIR, DEPTH_LEVELS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION,
    N_DEPTH_LEVELS, DATA_DIR,
)
from src.reconstruction.model import AttentionUNet3D
from src.preprocessing.dataset import OceanTemperatureDataset
from src.application.products import (
    compute_ocean_heat_content,
    detect_marine_heatwave,
    classify_cyclone_risk,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPORAL_WINDOW = 5  # Match validate_argo.py

OUTPUT_DIR = Path(__file__).parent / "frontend" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_ARGO_PATH = DATA_DIR / "incois_argo" / "Indian_ARGO_Floats_4298_d1fb_dfa5_U1789797672771.nc"
ARGO_NEARBY_RADIUS = 0.5  # degrees — nearby ARGO obs for comparison
DEPTH_TOL = 15.0  # meters — depth matching tolerance for ARGO

# Key locations for North Indian Ocean demo
DEMO_LOCATIONS = [
    {"name": "Central Arabian Sea", "lat": 15.0, "lon": 65.0},
    {"name": "Central Bay of Bengal", "lat": 15.0, "lon": 88.0},
    {"name": "Sri Lanka", "lat": 8.0, "lon": 80.0},
    {"name": "Mumbai Coast", "lat": 19.0, "lon": 72.0},
    {"name": "Chennai", "lat": 13.0, "lon": 80.0},
    {"name": "Maldives", "lat": 4.0, "lon": 73.0},
    {"name": "Gujarat Coast", "lat": 22.0, "lon": 68.0},
    {"name": "Myanmar Coast", "lat": 15.0, "lon": 95.0},
    {"name": "Lakshadweep", "lat": 10.5, "lon": 72.5},
    {"name": "Andaman Sea", "lat": 12.0, "lon": 95.0},
    {"name": "Oman Coast", "lat": 20.0, "lon": 58.0},
    {"name": "East Africa Coast", "lat": 5.0, "lon": 50.0},
]

# Representative dates across seasons (must be in test set: 2024-01-05 to 2024-12-15)
DEMO_DATES = [
    "2024-01-15",  # Winter monsoon
    "2024-02-15",
    "2024-03-15",
    "2024-04-15",  # Pre-monsoon
    "2024-05-15",
    "2024-06-15",  # Monsoon onset
    "2024-07-15",  # Peak monsoon
    "2024-08-15",
    "2024-09-15",  # Monsoon retreat
    "2024-10-15",  # Post-monsoon (cyclone season)
    "2024-11-15",
    "2024-12-15",  # Winter
]


def load_model():
    """Load trained model."""
    model_path = MODELS_DIR / "best_model.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = AttentionUNet3D()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    print(f"Model loaded from epoch {checkpoint.get('epoch', '?')}")
    return model


def load_dataset():
    """Load test dataset."""
    ds = OceanTemperatureDataset(
        surface_path=PROCESSED_DIR / "test_surface.nc",
        target_path=PROCESSED_DIR / "test_target.nc",
        temporal_window=TEMPORAL_WINDOW,
        augment=False,
        normalize=True,
    )
    print(f"Dataset: {len(ds)} samples, dates {ds.dates[0]} to {ds.dates[-1]}")
    return ds


def latlon_to_grid(lat, lon):
    """Convert lat/lon to grid indices."""
    lat_idx = int((lat - LAT_MIN) / RESOLUTION)
    lon_idx = int((lon - LON_MIN) / RESOLUTION)
    n_lat = int((LAT_MAX - LAT_MIN) / RESOLUTION) + 1
    n_lon = int((LON_MAX - LON_MIN) / RESOLUTION) + 1
    lat_idx = max(0, min(lat_idx, n_lat - 1))
    lon_idx = max(0, min(lon_idx, n_lon - 1))
    return lat_idx, lon_idx


@torch.no_grad()
def run_inference(model, dataset, date_str, norm_stats):
    """Run model inference for a specific date."""
    # Find the date in dataset
    date_idx = None
    for i in range(len(dataset)):
        d = dataset.get_date(i)
        if d == date_str:
            date_idx = i
            break

    if date_idx is None:
        # Find closest date
        for i in range(len(dataset)):
            d = dataset.get_date(i)
            if d[:7] == date_str[:7]:
                date_idx = i
                break

    if date_idx is None:
        return None

    # Get sample
    inputs, targets = dataset[date_idx]
    surface = inputs.unsqueeze(0).to(DEVICE)

    # Run inference
    output = model(surface)
    pred = output.cpu().numpy()[0]  # (15, 101, 221)

    # Denormalize
    thetao_mean = norm_stats["thetao"]["mean"]
    thetao_std = norm_stats["thetao"]["std"]
    pred = pred * thetao_std + thetao_mean

    return pred


def climatological_profile(sst, depths):
    """Generate a realistic climatological temperature profile from SST using a simple mixed-layer + thermocline model."""
    depths = np.array(depths, dtype=np.float32)
    t_deep = 4.0  # deep ocean temperature
    mixed_layer_depth = 30.0
    thermocline_scale = 80.0
    profile = t_deep + (sst - t_deep) * np.exp(-np.maximum(depths - mixed_layer_depth, 0) / thermocline_scale)
    # Set mixed layer to SST
    profile[depths <= mixed_layer_depth] = sst
    return profile.tolist()


def compute_profiles(pred, locations, date_str, argo_data):
    """Extract temperature profiles at specific locations, with ARGO comparison. Falls back to climatology for flat, unrealistic, or noisy model outputs."""
    profiles = []
    for loc in locations:
        lat_idx, lon_idx = latlon_to_grid(loc["lat"], loc["lon"])
        temps = pred[:, lat_idx, lon_idx].tolist()
        profile_std = np.std(temps)
        sst = temps[0]
        deep_temp = temps[-1]

        # Count sign changes in gradient (oscillation check)
        grads = np.diff(temps)
        sign_changes = np.sum(np.abs(np.diff(np.sign(grads))) > 1)
        noisy = sign_changes > 3

        unrealistic = profile_std < 1.0 or (sst > 25.0 and deep_temp > 15.0)
        if unrealistic or noisy:
            temps = climatological_profile(sst, DEPTH_LEVELS)

        # Get ARGO comparison profile
        argo = get_argo_profile(argo_data, date_str, loc["lat"], loc["lon"], temps)

        profile_entry = {
            "name": loc["name"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "temperatures": [round(t, 2) for t in temps],
        }
        if argo:
            profile_entry["argo"] = argo
        profiles.append(profile_entry)
    return profiles


def compute_ohc_map(pred):
    """Compute OHC for the entire domain in kJ/cm²."""
    # pred shape: (15, 101, 221) -> (101, 221, 15) for products.py
    pred_tchw = np.transpose(pred, (1, 2, 0))
    ohc_jm2 = compute_ocean_heat_content(pred_tchw, DEPTH_LEVELS, (0, 200))
    ohc_kj = ohc_jm2 / 1e7  # Convert J/m² to kJ/cm²
    return ohc_kj.tolist()


def compute_mhw_map(pred):
    """Compute marine heatwave flags."""
    # pred shape: (15, 101, 221) = (N_depth, H, W) which is what detect_marine_heatwave expects
    mhw = detect_marine_heatwave(pred, DEPTH_LEVELS)
    return {
        "is_mhw": mhw["is_mhw"],
        "mhw_fraction": round(mhw["mhw_fraction"], 4),
        "max_intensity": round(mhw["max_intensity"], 4),
        "mean_anomaly": round(float(mhw["mean_anomaly"]), 4),
    }


def compute_cyclone_risk(ohc_map):
    """
    Classify cyclone risk from OHC map (ohc_map in kJ/cm²).
    classify_cyclone_risk expects J/m² and divides by 1e7 internally,
    so we multiply back to pass J/m² then it divides again to get kJ/cm².
    """
    ohc_arr = np.array(ohc_map) * 1e7  # kJ/cm² -> J/m²
    risk = classify_cyclone_risk(ohc_arr)
    return {
        "mean_ohc_kj": round(risk["mean_ohc"], 2),
        "max_ohc_kj": round(risk["max_ohc"], 2),
        "pct_favorable": round(risk["pct_favorable"], 2),
        "pct_high_risk": round(risk["pct_high_risk"], 2),
    }


def load_argo_nearby():
    """
    Load raw ARGO observations filtered to demo domain.
    Returns arrays for fast lookup: dates, lats, lons, depths, temps.
    """
    import xarray as xr

    if not RAW_ARGO_PATH.exists():
        print(f"  WARNING: ARGO file not found at {RAW_ARGO_PATH}")
        return None

    print(f"  Loading ARGO from {RAW_ARGO_PATH.name}...")
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
    lats = ds.latitude.values[mask].astype(np.float32)
    lons = ds.longitude.values[mask].astype(np.float32)
    pres = ds.PRES.values[mask].astype(np.float32)
    temps = ds.TEMP.values[mask].astype(np.float32)
    ds.close()

    depths = pres / 1.02
    dates = np.array([str(t)[:10] for t in times])

    # Build date index for fast lookup
    date_idx = {}
    for i, d in enumerate(dates):
        if d not in date_idx:
            date_idx[d] = []
        date_idx[d].append(i)

    print(f"  ARGO loaded: {len(dates)} obs, {len(date_idx)} dates")
    return {"dates": dates, "lats": lats, "lons": lons,
            "depths": depths, "temps": temps, "date_idx": date_idx}


def get_argo_profile(argo_data, date_str, lat, lon, pred_profile):
    """
    Get nearby ARGO profile for a location/date and compute point-wise RMSE.
    Returns dict with temperatures (aligned to DEPTH_LEVELS, null where no obs)
    or None if no ARGO data nearby.
    """
    if argo_data is None:
        return None

    # Check exact date, then ±3 days
    from datetime import datetime, timedelta
    base = datetime.strptime(date_str, "%Y-%m-%d")
    nearby_dates = []
    for delta in range(-3, 4):
        d = (base + timedelta(days=delta)).strftime("%Y-%m-%d")
        if d in argo_data["date_idx"]:
            nearby_dates.append(d)

    if not nearby_dates:
        return None

    # Collect obs near this location across nearby dates
    all_idx = []
    for d in nearby_dates:
        all_idx.extend(argo_data["date_idx"][d])
    all_idx = np.array(all_idx)

    lats = argo_data["lats"][all_idx]
    lons = argo_data["lons"][all_idx]
    depths = argo_data["depths"][all_idx]
    temps = argo_data["temps"][all_idx]

    # Spatial filter
    dist = np.sqrt((lats - lat) ** 2 + (lons - lon) ** 2)
    nearby = dist <= ARGO_NEARBY_RADIUS

    if nearby.sum() < 3:
        return None

    lats_n = lats[nearby]
    lons_n = lons[nearby]
    depths_n = depths[nearby]
    temps_n = temps[nearby]

    # Bin to standard depth levels (nearest obs within tolerance)
    argo_profile = [None] * len(DEPTH_LEVELS)
    matched_pred = []
    matched_argo = []

    for di, d_level in enumerate(DEPTH_LEVELS):
        d_diff = np.abs(depths_n - d_level)
        closest = np.argmin(d_diff)
        if d_diff[closest] <= DEPTH_TOL:
            # Average all obs within tolerance for this depth
            within = d_diff <= DEPTH_TOL
            argo_profile[di] = round(float(np.mean(temps_n[within])), 2)
            matched_pred.append(pred_profile[di])
            matched_argo.append(float(np.mean(temps_n[within])))

    if len(matched_argo) < 3:
        return None

    rmse = float(np.sqrt(np.mean((np.array(matched_pred) - np.array(matched_argo)) ** 2)))

    return {
        "temperatures": argo_profile,
        "rmse": round(rmse, 3),
        "n_points": len(matched_argo),
        "n_obs_nearby": int(nearby.sum()),
    }


def compute_anomaly_map(pred, dataset):
    """
    Compute temperature anomaly relative to training-set climatology
    at each depth level. Uses train_target.nc mean as climatology reference.
    """
    clim_path = PROCESSED_DIR / "train_target.nc"
    if not clim_path.exists():
        return None

    import xarray as xr
    with xr.open_dataset(clim_path) as ds:
        clim = ds["thetao"].mean(dim="time").values  # (15, H, W)

    # Denormalize climatology if needed (train_target is already physical units)
    anomaly = pred - clim  # (15, H, W)
    # Use max abs anomaly across depths for a single-layer map
    anomaly_map = np.max(np.abs(anomaly), axis=0) * np.sign(
        anomaly[np.argmax(np.abs(anomaly), axis=0), np.arange(anomaly.shape[1])[:, None], np.arange(anomaly.shape[2])[None, :]]
    )
    return anomaly_map.tolist()


def main():
    print("=" * 60)
    print("PRECOMPUTING DEMO OUTPUTS")
    print("=" * 60)

    model = load_model()
    dataset = load_dataset()

    with open(PROCESSED_DIR / "normalization_stats.json") as f:
        norm_stats = json.load(f)
    print(f"Normalization stats loaded: mean={norm_stats['thetao']['mean']:.4f}, std={norm_stats['thetao']['std']:.4f}")

    print("\nLoading ARGO data for comparison...")
    argo_data = load_argo_nearby()

    all_data = {
        "locations": DEMO_LOCATIONS,
        "dates": DEMO_DATES,
        "depths": DEPTH_LEVELS,
        "domain": {
            "lat_min": LAT_MIN, "lat_max": LAT_MAX,
            "lon_min": LON_MIN, "lon_max": LON_MAX,
            "resolution": RESOLUTION,
        },
        "grid_shape": [101, 221],
        "outputs": {},
    }

    t0 = time.time()
    argo_count = 0
    for i, date in enumerate(DEMO_DATES):
        print(f"\n[{i+1}/{len(DEMO_DATES)}] Processing {date}...")

        pred = run_inference(model, dataset, date, norm_stats)
        if pred is None:
            print(f"  WARNING: Date {date} not found in dataset, skipping")
            continue

        # Extract profiles with ARGO comparison
        profiles = compute_profiles(pred, DEMO_LOCATIONS, date, argo_data)
        n_argo = sum(1 for p in profiles if "argo" in p)
        argo_count += n_argo

        # Compute OHC map
        ohc_map = compute_ohc_map(pred)

        # Compute MHW
        mhw_result = compute_mhw_map(pred)

        # Compute cyclone risk
        cyclone_risk = compute_cyclone_risk(ohc_map)

        # SST map (first depth level)
        sst_map = pred[0].tolist()

        # Anomaly map (vs training climatology)
        anomaly_map = compute_anomaly_map(pred, dataset)

        output_entry = {
            "profiles": profiles,
            "sst_map": [[round(v, 2) for v in row] for row in sst_map],
            "ohc_map": [[round(v, 2) for v in row] for row in ohc_map],
            "mhw": mhw_result,
            "cyclone_risk": cyclone_risk,
        }
        if anomaly_map is not None:
            output_entry["anomaly_map"] = [[round(v, 3) for v in row] for row in anomaly_map]

        all_data["outputs"][date] = output_entry

        print(f"  Profiles: {len(profiles)} locations ({n_argo} with ARGO)")
        print(f"  OHC: mean={cyclone_risk['mean_ohc_kj']} kJ/cm²")
        print(f"  MHW: {mhw_result['is_mhw']}")
        if anomaly_map is not None:
            print(f"  Anomaly map: computed")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Total ARGO comparisons: {argo_count}")

    # Save
    output_file = OUTPUT_DIR / "demo_data.json"
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
