"""
Real-time data fetcher for OceanEmbed online demo.
Fetches current SST, wind, and sea level from Open-Meteo Marine API (free, no key).
"""
import numpy as np
import requests
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Open-Meteo APIs (free, no API key)
MARINE_API = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

# Domain: North Indian Ocean
LAT_MIN, LAT_MAX = 0.0, 25.0
LON_MIN, LON_MAX = 45.0, 100.0
RESOLUTION = 0.25
N_LAT, N_LON = 101, 221

# Grid points for API sampling (sparse grid, will interpolate)
SAMPLE_LATS = np.arange(LAT_MIN + 1, LAT_MAX, 3)
SAMPLE_LONS = np.arange(LON_MIN + 2, LON_MAX, 4)


def fetch_marine_data(lat: float, lon: float, date_str: str = None) -> dict:
    """Fetch current marine conditions from Open-Meteo Marine API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "sea_surface_temperature,sea_level_height_msl,ocean_current_velocity,ocean_current_direction",
        "timezone": "GMT",
    }
    if date_str:
        params["start_date"] = date_str
        params["end_date"] = date_str
        params["hourly"] = "sea_surface_temperature,sea_level_height_msl"
    
    try:
        resp = requests.get(MARINE_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        logger.warning("Marine API failed for (%.1f, %.1f): %s", lat, lon, e)
        return None


def fetch_weather_data(lat: float, lon: float) -> dict:
    """Fetch current weather (wind) from Open-Meteo Weather API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "wind_speed_10m,wind_direction_10m",
        "timezone": "GMT",
    }
    try:
        resp = requests.get(WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Weather API failed for (%.1f, %.1f): %s", lat, lon, e)
        return None


def fetch_sst_grid() -> np.ndarray:
    """
    Fetch real-time SST across the North Indian Ocean domain.
    Uses Open-Meteo Marine API at sparse grid points, then interpolates.
    Returns: (101, 221) SST array in Celsius
    """
    logger.info("Fetching real-time SST from Open-Meteo Marine API...")
    
    sst_sparse = {}
    lats_sparse = []
    lons_sparse = []
    sst_values = []
    
    total = len(SAMPLE_LATS) * len(SAMPLE_LONS)
    count = 0
    
    for lat in SAMPLE_LATS:
        for lon in SAMPLE_LONS:
            count += 1
            if count % 20 == 0:
                logger.info("  SST fetch: %d/%d grid points", count, total)
            
            data = fetch_marine_data(lat, lon)
            if data and "current" in data:
                sst = data["current"].get("sea_surface_temperature")
                if sst is not None:
                    sst_sparse[(lat, lon)] = sst
                    lats_sparse.append(lat)
                    lons_sparse.append(lon)
                    sst_values.append(sst)
    
    if not sst_values:
        logger.warning("No SST data retrieved, using climatological fallback")
        return generate_climatological_sst()
    
    logger.info("Retrieved SST at %d/%d points", len(sst_values), total)
    
    # Interpolate to full grid
    from scipy.interpolate import griddata
    
    lats_full = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lons_full = np.linspace(LON_MIN, LON_MAX, N_LON)
    
    grid_lats, grid_lons = np.meshgrid(lats_full, lons_full, indexing='ij')
    
    points = np.column_stack([lats_sparse, lons_sparse])
    sst_grid = griddata(points, sst_values, (grid_lats, grid_lons), method='cubic')
    
    # Fill NaN with nearest
    mask = np.isnan(sst_grid)
    if mask.any():
        sst_nearest = griddata(points, sst_values, (grid_lats, grid_lons), method='nearest')
        sst_grid[mask] = sst_nearest[mask]
    
    return sst_grid


def fetch_wind_grid() -> tuple:
    """
    Fetch real-time wind across the domain.
    Returns: (wind_u, wind_v) each (101, 221) in m/s
    """
    logger.info("Fetching real-time wind from Open-Meteo Weather API...")
    
    wind_u_sparse = {}
    wind_v_sparse = {}
    
    total = len(SAMPLE_LATS) * len(SAMPLE_LONS)
    count = 0
    
    for lat in SAMPLE_LATS:
        for lon in SAMPLE_LONS:
            count += 1
            if count % 20 == 0:
                logger.info("  Wind fetch: %d/%d grid points", count, total)
            
            data = fetch_weather_data(lat, lon)
            if data and "current" in data:
                ws = data["current"].get("wind_speed_10m", 0)
                wd = data["current"].get("wind_direction_10m", 0)
                # Convert km/h to m/s and decompose
                ws_ms = ws / 3.6
                wd_rad = np.radians(wd)
                wind_u_sparse[(lat, lon)] = -ws_ms * np.sin(wd_rad)  # East component
                wind_v_sparse[(lat, lon)] = -ws_ms * np.cos(wd_rad)  # North component
    
    if not wind_u_sparse:
        logger.warning("No wind data retrieved, using calm conditions")
        return np.zeros((N_LAT, N_LON)), np.zeros((N_LAT, N_LON))
    
    from scipy.interpolate import griddata
    
    lats_full = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lons_full = np.linspace(LON_MIN, LON_MAX, N_LON)
    grid_lats, grid_lons = np.meshgrid(lats_full, lons_full, indexing='ij')
    
    points = np.column_stack([
        [k[0] for k in wind_u_sparse.keys()],
        [k[1] for k in wind_u_sparse.keys()]
    ])
    u_vals = list(wind_u_sparse.values())
    v_vals = list(wind_v_sparse.values())
    
    wind_u = griddata(points, u_vals, (grid_lats, grid_lons), method='nearest')
    wind_v = griddata(points, v_vals, (grid_lats, grid_lons), method='nearest')
    
    return wind_u, wind_v


def generate_climatological_sst() -> np.ndarray:
    """Generate climatological SST for North Indian Ocean (fallback)."""
    lats = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lons = np.linspace(LON_MIN, LON_MAX, N_LON)
    
    month = datetime.now().month
    
    # Simple seasonal SST model for North Indian Ocean
    if month in [6, 7, 8, 9]:  # Monsoon
        base_temp = 27.5
    elif month in [12, 1, 2]:  # Winter
        base_temp = 25.0
    else:  # Pre/post monsoon
        base_temp = 28.0
    
    # Latitude gradient (warmer near equator)
    lat_effect = (lats[:, None] - 12.5) * (-0.08)
    
    # Add some spatial variation
    sst = base_temp + lat_effect + np.random.normal(0, 0.3, (N_LAT, N_LON))
    
    return sst


def generate_climatological_ssh() -> np.ndarray:
    """Generate climatological SSH (sea surface height anomaly)."""
    return np.random.normal(0.0, 0.15, (N_LAT, N_LON))


def generate_climatological_sss() -> np.ndarray:
    """Generate climatological SSS (sea surface salinity)."""
    lats = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    base_sss = 34.5 + (lats[:, None] - 12.5) * 0.02
    return base_sss + np.random.normal(0, 0.2, (N_LAT, N_LON))


def build_model_input(
    sst_grid: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
) -> np.ndarray:
    """
    Build 5-channel model input from real-time data.
    Returns: (5, 1, 101, 221) - channels x time x lat x lon
    Single timestep (temporal_window=1 for real-time)
    """
    ssh = generate_climatological_ssh()
    sss = generate_climatological_sss()
    
    # Normalize each channel (approximate normalization from training stats)
    # SST: mean~25, std~3
    sst_norm = (sst_grid - 25.0) / 3.0
    # SSH: mean~0, std~0.2
    ssh_norm = ssh / 0.2
    # SSS: mean~34.5, std~0.5
    sss_norm = (sss - 34.5) / 0.5
    # Wind: mean~0, std~3
    wind_u_norm = wind_u / 3.0
    wind_v_norm = wind_v / 3.0
    
    # Stack channels: (5, 101, 221)
    input_2d = np.stack([sst_norm, ssh_norm, sss_norm, wind_u_norm, wind_v_norm], axis=0)
    
    # Add time dimension: (5, 1, 101, 221)
    return input_2d[:, np.newaxis, :, :]


def fetch_realtime_data() -> dict:
    """
    Fetch all real-time data and prepare model input.
    Returns dict with grids and metadata.
    """
    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    
    logger.info("Fetching real-time ocean data for %s", date_str)
    
    sst = fetch_sst_grid()
    wind_u, wind_v = fetch_wind_grid()
    
    model_input = build_model_input(sst, wind_u, wind_v)
    
    return {
        "date": date_str,
        "timestamp": now.isoformat(),
        "sst": sst,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "model_input": model_input,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_realtime_data()
    print(f"Date: {result['date']}")
    print(f"SST range: {result['sst'].min():.1f} - {result['sst'].max():.1f} °C")
    print(f"Model input shape: {result['model_input'].shape}")
