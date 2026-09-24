"""
Utility functions for OceanEmbed.
Includes visualization helpers, data I/O, and common operations.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.config import DEPTH_LEVELS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


def plot_temperature_section(
    temperature: np.ndarray,
    latitudes: np.ndarray,
    depths: list[int] = None,
    title: str = "Temperature Section",
    save_path: Optional[Path] = None,
    vmin: float = 0,
    vmax: float = 30,
):
    """Plot a temperature cross-section (depth vs latitude/longitude)."""
    depths = depths or DEPTH_LEVELS

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.pcolormesh(
        latitudes, depths, temperature,
        cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
        shading="auto",
    )

    ax.set_xlabel("Latitude (°N)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()

    cbar = plt.colorbar(im, ax=ax, label="Temperature (°C)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_map_overlay(
    data: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    variable_name: str = "SST",
    title: str = "Sea Surface Temperature",
    save_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdYlBu_r",
):
    """Plot a 2D map overlay."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if vmin is None:
        vmin = np.nanpercentile(data, 5)
    if vmax is None:
        vmax = np.nanpercentile(data, 95)

    im = ax.pcolormesh(
        longitudes, latitudes, data,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto",
    )

    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    plt.colorbar(im, ax=ax, label=variable_name)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_error_by_depth(
    depth_rmse: dict,
    title: str = "RMSE by Depth Level",
    save_path: Optional[Path] = None,
):
    """Plot RMSE breakdown by depth level."""
    depths = list(depth_rmse.keys())
    rmses = [depth_rmse[d]["rmse"] for d in depths]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(depths, rmses, color="#4fc3f7", edgecolor="#1a3a5c")
    ax.set_xlabel("RMSE (°C)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def plot_validation_comparison(
    argo_profiles: np.ndarray,
    model_profiles: np.ndarray,
    depths: list[int] = None,
    n_examples: int = 4,
    save_path: Optional[Path] = None,
):
    """Plot ARGO vs model-reconstructed profiles side by side."""
    depths = depths or DEPTH_LEVELS

    fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 6))
    if n_examples == 1:
        axes = [axes]

    indices = np.random.choice(len(argo_profiles), min(n_examples, len(argo_profiles)), replace=False)

    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.plot(argo_profiles[idx], depths, "o-", color="#ff9800", label="ARGO", markersize=3)
        ax.plot(model_profiles[idx], depths, "s--", color="#4fc3f7", label="Model", markersize=3)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("ARGO vs Reconstructed Profiles", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig


def save_netcdf_output(
    temperature: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    depths: list[int],
    dates: list[str],
    output_path: Path,
    variable_name: str = "temperature",
):
    """Save reconstruction output as NetCDF."""
    ds = xr.Dataset(
        {
            variable_name: (["time", "depth", "latitude", "longitude"], temperature),
        },
        coords={
            "time": dates,
            "depth": depths,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )
    ds[variable_name].attrs["units"] = "celsius"
    ds[variable_name].attrs["long_name"] = "Reconstructed ocean temperature"
    ds.to_netcdf(output_path)
    return ds


def load_scheduled_dataset(path: Path) -> xr.Dataset:
    """Load a dataset with time coordinate handling."""
    ds = xr.open_dataset(path)
    if "time" in ds.dims:
        ds = ds.sortby("time")
    return ds
