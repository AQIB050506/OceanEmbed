"""
Application layer for OceanEmbed.
Derives decision-relevant signals from reconstructed temperature fields:
1. Ocean Heat Content (OHC) — relevant to cyclone intensification
2. Marine heatwave / thermal anomaly detection
"""
import numpy as np
from typing import Optional
import logging

from src.config import DEPTH_LEVELS

logger = logging.getLogger(__name__)

RHO_SW = 1025.0    # Seawater density (kg/m³)
CP_SW = 3985.0     # Specific heat capacity of seawater (J/kg/°C)


def compute_ocean_heat_content(
    temperature: np.ndarray,
    depths: list[int] = None,
    depth_range: tuple[int, int] = (0, 200),
) -> np.ndarray:
    """
    Compute Ocean Heat Content (OHC) from temperature profiles.

    OHC = rho * cp * integral(T(z) dz) over specified depth range

    Args:
        temperature: Temperature array, shape (..., N_depth)
                     Last dimension corresponds to depth levels
        depths: List of depth values in meters
        depth_range: (min_depth, max_depth) for OHC integration

    Returns:
        OHC array in J/m²
    """
    depths = depths or DEPTH_LEVELS
    depths = np.array(depths, dtype=float)

    min_d, max_d = depth_range
    mask = (depths >= min_d) & (depths <= max_d)
    sel_depths = depths[mask]

    if temperature.ndim == 1:
        sel_temp = temperature[mask]
    elif temperature.ndim == 2:
        sel_temp = temperature[:, mask]
    elif temperature.ndim == 3:
        sel_temp = temperature[:, :, mask]
    elif temperature.ndim == 4:
        sel_temp = temperature[:, :, :, mask]
    else:
        sel_temp = temperature[..., mask]

    ohc = np.zeros(sel_temp.shape[:-1])

    for i in range(len(sel_depths) - 1):
        dz = sel_depths[i + 1] - sel_depths[i]
        t_avg = 0.5 * (sel_temp[..., i] + sel_temp[..., i + 1])
        ohc += RHO_SW * CP_SW * t_avg * dz

    return ohc


def compute_ohc_anomaly(
    ohc_current: np.ndarray,
    ohc_climatology: np.ndarray,
) -> np.ndarray:
    """
    Compute OHC anomaly relative to climatological mean.

    Args:
        ohc_current: Current OHC (H, W) or (T, H, W)
        ohc_climatology: Climatological mean OHC (H, W)

    Returns:
        OHC anomaly in J/m²
    """
    return ohc_current - ohc_climatology


def detect_marine_heatwave(
    temperature: np.ndarray,
    depths: list[int] = None,
    climatology_mean: Optional[np.ndarray] = None,
    climatology_std: Optional[np.ndarray] = None,
    threshold_percentile: float = 90,
    depth_range: tuple[int, int] = (0, 100),
) -> dict:
    """
    Detect marine heatwave conditions from subsurface temperature.

    A marine heatwave is defined as when subsurface temperature exceeds
    the 90th percentile of the climatological distribution.

    Args:
        temperature: Temperature profile (N_depth, H, W) or (H, W) per depth
        depths: Depth levels
        climatology_mean: Mean climatology (N_depth, H, W)
        climatology_std: Std of climatology (N_depth, H, W)
        threshold_percentile: Percentile threshold for MHW detection
        depth_range: Depth range to assess

    Returns:
        Dictionary with MHW flags and intensity
    """
    depths = depths or DEPTH_LEVELS
    depths_arr = np.array(depths)
    min_d, max_d = depth_range
    depth_mask = (depths_arr >= min_d) & (depths_arr <= max_d)
    n_depths_total = len(depths_arr)

    if temperature.ndim == 2:
        temperature = temperature[np.newaxis, :, :]

    sel_temp = temperature[depth_mask]

    if climatology_mean is None:
        climatology_mean = np.mean(temperature, axis=0, keepdims=True)
    if climatology_std is None:
        climatology_std = np.std(temperature, axis=0, keepdims=True) + 1e-8

    if climatology_mean.shape[0] == n_depths_total:
        sel_mean = climatology_mean[depth_mask]
    else:
        sel_mean = climatology_mean

    if climatology_std.shape[0] == n_depths_total:
        sel_std = climatology_std[depth_mask]
    else:
        sel_std = climatology_std

    anomaly = sel_temp - sel_mean
    normalized = anomaly / sel_std

    mhw_flag = normalized > 1.28

    intensity = np.clip(normalized, 0, None)

    spatial_mhw = np.any(mhw_flag, axis=0)
    max_intensity = np.max(intensity, axis=0)

    return {
        "is_mhw": bool(spatial_mhw.any()),
        "mhw_fraction": float(spatial_mhw.mean()),
        "spatial_flag": spatial_mhw,
        "max_intensity": float(max_intensity.mean()),
        "mean_anomaly": float(anomaly.mean()),
        "depth_max_anomaly": float(depths_arr[depth_mask][np.argmax(np.abs(anomaly.mean(axis=tuple(range(1, anomaly.ndim)))))]) if anomaly.size > 0 else None,
    }


def classify_cyclone_risk(ohc: np.ndarray) -> dict:
    """
    Classify cyclone intensification risk based on OHC (0-200m).

    Thresholds calibrated for North Indian Ocean absolute OHC (kJ/cm²):
    - OHC > 2000 kJ/cm²: Favorable for cyclone intensification
    - OHC > 2200 kJ/cm²: Highly favorable (rapid intensification risk)
    - OHC > 2400 kJ/cm²: Extreme risk

    Args:
        ohc: Ocean Heat Content in J/m²

    Returns:
        Risk classification dictionary
    """
    ohc_kj_cm2 = ohc / 1e7  # Convert J/m² to kJ/cm²

    risk_level = np.zeros_like(ohc_kj_cm2, dtype=int)
    risk_level[ohc_kj_cm2 > 2000] = 1
    risk_level[ohc_kj_cm2 > 2200] = 2
    risk_level[ohc_kj_cm2 > 2400] = 3

    return {
        "risk_level": risk_level,
        "ohc_kj_cm2": ohc_kj_cm2,
        "mean_ohc": float(ohc_kj_cm2.mean()),
        "max_ohc": float(ohc_kj_cm2.max()),
        "pct_favorable": float((ohc_kj_cm2 > 2000).mean()) * 100,
        "pct_high_risk": float((ohc_kj_cm2 > 2200).mean()) * 100,
    }


def generate_application_products(
    temperature: np.ndarray,
    depths: list[int] = None,
    latitudes: Optional[np.ndarray] = None,
    longitudes: Optional[np.ndarray] = None,
) -> dict:
    """
    Generate all application layer products from reconstructed temperature.

    Args:
        temperature: 3D temperature field (N_depth, H, W)
        depths: Depth levels
        latitudes: Latitude array
        longitudes: Longitude array

    Returns:
        Dictionary of application products
    """
    depths = depths or DEPTH_LEVELS

    ohc_0_200 = compute_ocean_heat_content(temperature, depths, (0, 200))
    ohc_0_100 = compute_ocean_heat_content(temperature, depths, (0, 100))

    mhw = detect_marine_heatwave(temperature, depths)

    cyclone_risk = classify_cyclone_risk(ohc_0_200)

    mixed_layer_temp = temperature[0] if len(temperature) > 0 else None
    thermocline_depth_idx = np.argmin(np.abs(np.array(depths) - 100))
    thermocline_temp = temperature[thermocline_depth_idx] if len(temperature) > thermocline_depth_idx else None

    return {
        "ocean_heat_content_200m": ohc_0_200,
        "ocean_heat_content_100m": ohc_0_100,
        "marine_heatwave": mhw,
        "cyclone_risk": cyclone_risk,
        "surface_temperature": mixed_layer_temp,
        "thermocline_temperature": thermocline_temp,
    }
