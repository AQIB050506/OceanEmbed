"""
Real-time data fetcher for OceanEmbed online demo.
Fetches current SST, wind from Open-Meteo APIs (free, no key),
builds a properly-normalized (5, 5, 101, 221) model input.
"""
import numpy as np
import requests
import json
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION

logger = logging.getLogger(__name__)

MARINE_API = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

N_LAT, N_LON = 101, 221
TEMPORAL_WINDOW = 5

# Sparse sample grid — interpolated to full domain after fetch
SAMPLE_LATS = np.arange(LAT_MIN + 1.5, LAT_MAX, 4.0)   # ~6 points
SAMPLE_LONS = np.arange(LON_MIN + 2.0, LON_MAX, 5.0)   # ~11 points

# Load training normalization stats once
_STATS_PATH = PROCESSED_DIR / "normalization_stats.json"
with open(_STATS_PATH) as f:
    NORM_STATS = json.load(f)


def _fetch_marine(lat, lon, timeout=8):
    try:
        r = requests.get(MARINE_API, params={
            "latitude": float(lat), "longitude": float(lon),
            "current": "sea_surface_temperature,sea_level_height_msl,ocean_current_velocity,ocean_current_direction",
            "timezone": "GMT",
        }, timeout=timeout)
        r.raise_for_status()
        return (lat, lon, r.json())
    except Exception as e:
        logger.debug("Marine (%.1f,%.1f): %s", lat, lon, e)
        return (lat, lon, None)


def _fetch_weather(lat, lon, timeout=8):
    try:
        r = requests.get(WEATHER_API, params={
            "latitude": float(lat), "longitude": float(lon),
            "current": "wind_speed_10m,wind_direction_10m",
            "timezone": "GMT",
        }, timeout=timeout)
        r.raise_for_status()
        return (lat, lon, r.json())
    except Exception as e:
        logger.debug("Weather (%.1f,%.1f): %s", lat, lon, e)
        return (lat, lon, None)


def _interp_to_grid(points, values, method="cubic"):
    from scipy.interpolate import griddata
    lats_f = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lons_f = np.linspace(LON_MIN, LON_MAX, N_LON)
    glat, glon = np.meshgrid(lats_f, lons_f, indexing="ij")
    pts = np.asarray(points, dtype=np.float64)  # (N, 2)
    vals = np.asarray(values, dtype=np.float64)
    if method == "cubic" and len(vals) < 4:
        method = "linear"
    if method == "linear" and len(vals) < 3:
        method = "nearest"
    grid = griddata(pts, vals, (glat, glon), method=method)
    mask = np.isnan(grid)
    if mask.any():
        nearest = griddata(pts, vals, (glat, glon), method="nearest")
        grid[mask] = nearest[mask]
    return grid


def fetch_sst_grid(max_workers=12):
    """Fetch SST and SSH together (same marine API call). Returns (sst, ssh, source)."""
    coords = [(la, lo) for la in SAMPLE_LATS for lo in SAMPLE_LONS]
    logger.info("Fetching SST/SSH at %d points (%d workers)...", len(coords), max_workers)
    sst_pts, sst_vals = [], []
    ssh_pts, ssh_vals = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_fetch_marine, la, lo) for la, lo in coords]
        for f in as_completed(futs):
            lat, lon, data = f.result()
            if data and "current" in data:
                cur = data["current"]
                sst = cur.get("sea_surface_temperature")
                ssh = cur.get("sea_level_height_msl")
                if sst is not None:
                    sst_pts.append((lat, lon))
                    sst_vals.append(float(sst))
                if ssh is not None:
                    ssh_pts.append((lat, lon))
                    ssh_vals.append(float(ssh))
    if len(sst_vals) < 4:
        logger.warning("Only %d SST points — falling back to climatology", len(sst_vals))
        return generate_climatological_sst(), generate_climatological_ssh(), "climatology"
    logger.info("SST: %d/%d points, SSH: %d points", len(sst_vals), len(coords), len(ssh_vals))
    sst_grid = _interp_to_grid(sst_pts, sst_vals)
    if len(ssh_vals) >= 4:
        # Convert MSL (geoid-referenced) to anomaly by removing spatial mean
        ssh_grid = _interp_to_grid(ssh_pts, ssh_vals)
        ssh_grid = ssh_grid - np.nanmean(ssh_grid)
    else:
        ssh_grid = generate_climatological_ssh()
    return sst_grid, ssh_grid, "open-meteo"


def fetch_wind_grid(max_workers=12):
    coords = [(la, lo) for la in SAMPLE_LATS for lo in SAMPLE_LONS]
    logger.info("Fetching wind at %d points...", len(coords))
    pts, u_vals, v_vals = [], [], []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_fetch_weather, la, lo) for la, lo in coords]
        for f in as_completed(futs):
            lat, lon, data = f.result()
            if data and "current" in data:
                ws = data["current"].get("wind_speed_10m")
                wd = data["current"].get("wind_direction_10m")
                if ws is not None and wd is not None:
                    ws_ms = float(ws) / 3.6  # km/h -> m/s
                    wd_rad = np.radians(float(wd))
                    pts.append((lat, lon))
                    u_vals.append(-ws_ms * np.sin(wd_rad))
                    v_vals.append(-ws_ms * np.cos(wd_rad))
    if len(u_vals) < 4:
        logger.warning("Only %d wind points — using calm conditions", len(u_vals))
        return np.zeros((N_LAT, N_LON)), np.zeros((N_LAT, N_LON)), "calm"
    logger.info("Wind: %d/%d points retrieved", len(pts), len(coords))
    return (
        _interp_to_grid(pts, u_vals, method="nearest"),
        _interp_to_grid(pts, v_vals, method="nearest"),
        "open-meteo",
    )


def generate_climatological_sst():
    lats = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    month = datetime.now(timezone.utc).month
    if month in (6, 7, 8, 9):
        base = 27.5
    elif month in (12, 1, 2):
        base = 25.0
    else:
        base = 28.0
    lat_effect = (lats[:, None] - 12.5) * (-0.08)
    rng = np.random.default_rng(42)
    return base + lat_effect + rng.normal(0, 0.3, (N_LAT, N_LON))


def generate_climatological_ssh():
    """Smooth background SSH anomaly field (~0 m, small spatial structure)."""
    lats = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lons = np.linspace(LON_MIN, LON_MAX, N_LON)
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    # Large-scale pattern: higher SSH in central basin, lower near boundaries
    ssh = (
        0.05 * np.sin((glon - 50) * np.pi / 40.0)
        + 0.03 * np.cos((glat - 10) * np.pi / 25.0)
        + 0.02 * np.sin((glat + glon) * np.pi / 30.0)
    )
    return ssh


def generate_climatological_sss():
    """Smooth climatological SSS: fresher Bay of Bengal, saltier Arabian Sea."""
    lats = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lons = np.linspace(LON_MIN, LON_MAX, N_LON)
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    # Arabian Sea (~36-37) saltier than Bay of Bengal (~33-34)
    sss = 34.8 + 1.2 * np.tanh((glon - 78.0) / 12.0) * -1.0  # fresher east
    sss += -0.5 * np.exp(-((glat - 15) ** 2) / 50.0) * ((glon > 80).astype(float))  # BoB freshwater
    sss += 0.02 * (glat - 12.5)
    return sss


def build_model_input(sst_c, ssh, wind_u, wind_v):
    """
    Build (5, 5, 101, 221) normalized model input.
    Channels: sst, ssh, sss, wind_u, wind_v; Temporal: 5 days (replicated + small drift).
    SST from API is Celsius; training stats expect Kelvin.
    """
    sss = generate_climatological_sss()
    sst_k = sst_c + 273.15  # match training units (Kelvin)

    def norm(arr, var):
        mean = NORM_STATS[var]["mean"]
        std = NORM_STATS[var]["std"]
        return (arr - mean) / (std + 1e-8)

    sst_n = norm(sst_k, "sst")
    ssh_n = norm(ssh, "ssh")
    sss_n = norm(sss, "sss")
    u_n = norm(wind_u, "wind_u")
    v_n = norm(wind_v, "wind_v")

    channels = [sst_n, ssh_n, sss_n, u_n, v_n]
    input_arr = np.zeros((5, TEMPORAL_WINDOW, N_LAT, N_LON), dtype=np.float32)
    for t in range(TEMPORAL_WINDOW):
        drift = 0.02 * (t - TEMPORAL_WINDOW // 2)
        for c, ch in enumerate(channels):
            input_arr[c, t] = ch + drift * (0.01 if c == 0 else 0.001)
    return input_arr


def fetch_realtime_data():
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    logger.info("Fetching real-time ocean data for %s", date_str)
    sst, ssh, sst_src = fetch_sst_grid()
    wind_u, wind_v, wind_src = fetch_wind_grid()
    model_input = build_model_input(sst, ssh, wind_u, wind_v)

    return {
        "date": date_str,
        "timestamp": now.isoformat(),
        "sst": sst,
        "ssh": ssh,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "model_input": model_input,
        "sources": {"sst": sst_src, "wind": wind_src},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    result = fetch_realtime_data()
    print(f"Date: {result['date']}")
    print(f"SST range: {result['sst'].min():.1f} - {result['sst'].max():.1f} C ({result['sources']['sst']})")
    print(f"Wind source: {result['sources']['wind']}")
    print(f"Model input shape: {result['model_input'].shape}")
