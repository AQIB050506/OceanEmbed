"""
Data harmonization pipeline for OceanEmbed — FAST version.
Uses scipy.interpolate.RegularGridInterpolator for fast regridding.
Processes GLORYS year-by-year, chunked for memory efficiency.
"""
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from typing import Optional
import gc
import json
import logging
import time

from src.config import (
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION,
    DEPTH_LEVELS, N_DEPTH_LEVELS,
    GLORYS_DIR, SST_FILE, SSH_FILE, SSS_FILE, WIND_FILE,
    INTERIM_DIR, PROCESSED_DIR,
    TRAIN_START, TRAIN_END, VAL_START, TEST_START,
)

logger = logging.getLogger(__name__)

TARGET_LATS = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
TARGET_LONS = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)


def _fast_regrid_2d(src_lats, src_lons, data_2d, target_lats, target_lons):
    """Fast 2D regridding using scipy RegularGridInterpolator."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (src_lats, src_lons), data_2d,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    tg_lats, tg_lons = np.meshgrid(target_lats, target_lons, indexing="ij")
    return interp((tg_lats, tg_lons))


def regrid_dataset(ds, lat_name="latitude", lon_name="longitude"):
    """Regrid an entire xr.Dataset to TARGET_LATS x TARGET_LONS using scipy."""
    src_lats = ds[lat_name].values
    src_lons = ds[lon_name].values

    result = {}
    for var in ds.data_vars:
        da = ds[var]
        if lat_name not in da.dims or lon_name not in da.dims:
            result[var] = da
            continue

        shape = list(da.shape)
        lat_idx = da.dims.index(lat_name)
        lon_idx = da.dims.index(lon_name)

        new_shape = list(shape)
        new_shape[lat_idx] = len(TARGET_LATS)
        new_shape[lon_idx] = len(TARGET_LONS)

        out = np.empty(new_shape, dtype=np.float32)

        other_dims = [i for i in range(len(shape)) if i not in (lat_idx, lon_idx)]

        if len(other_dims) == 0:
            out[:] = _fast_regrid_2d(src_lats, src_lons, da.values, TARGET_LATS, TARGET_LONS)
        elif len(other_dims) == 1:
            t_idx = other_dims[0]
            n_steps = shape[t_idx]
            for t in range(n_steps):
                slab = da.isel({da.dims[t_idx]: t}).values
                out[t] = _fast_regrid_2d(src_lats, src_lons, slab, TARGET_LATS, TARGET_LONS)
        elif len(other_dims) == 2:
            d1_idx, d2_idx = other_dims
            for i in range(shape[d1_idx]):
                for j in range(shape[d2_idx]):
                    sel = {da.dims[d1_idx]: i, da.dims[d2_idx]: j}
                    slab = da.isel(sel).values
                    out[i, j] = _fast_regrid_2d(src_lats, src_lons, slab, TARGET_LATS, TARGET_LONS)
        else:
            for t in range(shape[other_dims[0]]):
                slab = da.isel({da.dims[other_dims[0]]: t}).values
                out[t] = _fast_regrid_2d(src_lats, src_lons, slab, TARGET_LATS, TARGET_LONS)

        dims_out = list(da.dims)
        dims_out[lat_idx] = "latitude"
        dims_out[lon_idx] = "longitude"

        result[var] = (dims_out, out, {})

    return xr.Dataset(result, coords={"latitude": TARGET_LATS, "longitude": TARGET_LONS})


def process_glorys_yearly(output_dir: Path) -> None:
    """Process GLORYS year-by-year using fast scipy regridding, chunked by time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    glorys_files = sorted(GLORYS_DIR.glob("GLORYS_NIO_thetao_so_*.nc"))

    if not glorys_files:
        logger.error("No GLORYS files found in %s", GLORYS_DIR)
        return

    logger.info("Processing %d GLORYS files with scipy regridding...", len(glorys_files))

    tg_lats, tg_lons = np.meshgrid(TARGET_LATS, TARGET_LONS, indexing="ij")

    for f in glorys_files:
        year = f.stem.split("_")[-1]
        out_file = output_dir / f"glorys_025_{year}.nc"

        if out_file.exists():
            logger.info("  %s already exists, skipping", out_file.name)
            continue

        t0 = time.time()
        logger.info("  Processing %s...", f.name)
        ds = xr.open_dataset(f)

        # Select depth levels first
        if "depth" in ds.dims:
            available_depths = ds.depth.values
            sel_idx, sel_depths = [], []
            for d in DEPTH_LEVELS:
                idx = np.argmin(np.abs(available_depths - d))
                if np.abs(available_depths[idx] - d) < 50:
                    sel_idx.append(idx)
                    sel_depths.append(d)
            ds = ds.isel(depth=sel_idx)
            ds = ds.assign_coords(depth=sel_depths)

        src_lats = ds.latitude.values
        src_lons = ds.longitude.values
        n_time = ds.sizes["time"]
        n_depth = ds.sizes["depth"]
        n_lat_out = len(TARGET_LATS)
        n_lon_out = len(TARGET_LONS)

        # Allocate output arrays
        thetao_out = np.empty((n_time, n_depth, n_lat_out, n_lon_out), dtype=np.float32)
        so_out = np.empty((n_time, n_depth, n_lat_out, n_lon_out), dtype=np.float32)

        # Process in time chunks to manage memory
        time_chunk = 10  # process 10 timesteps at a time
        for t_start in range(0, n_time, time_chunk):
            t_end = min(t_start + time_chunk, n_time)
            # Load only this time chunk
            thetao_chunk = ds["thetao"].isel(time=slice(t_start, t_end)).values  # (chunk, depth, lat, lon)
            so_chunk = ds["so"].isel(time=slice(t_start, t_end)).values

            for t_local in range(t_end - t_start):
                t_global = t_start + t_local
                for d in range(n_depth):
                    interp_t = RegularGridInterpolator(
                        (src_lats, src_lons), thetao_chunk[t_local, d],
                        method="linear", bounds_error=False, fill_value=np.nan,
                    )
                    thetao_out[t_global, d] = interp_t((tg_lats, tg_lons))

                    interp_s = RegularGridInterpolator(
                        (src_lats, src_lons), so_chunk[t_local, d],
                        method="linear", bounds_error=False, fill_value=np.nan,
                    )
                    so_out[t_global, d] = interp_s((tg_lats, tg_lons))

            del thetao_chunk, so_chunk
            gc.collect()
            if t_start % 50 == 0:
                logger.info("    GLORYS %s: %d/%d timesteps", year, t_end, n_time)

        result = xr.Dataset(
            {
                "thetao": (["time", "depth", "latitude", "longitude"], thetao_out),
                "so": (["time", "depth", "latitude", "longitude"], so_out),
            },
            coords={
                "time": ds.time.values,
                "depth": ds.depth.values,
                "latitude": TARGET_LATS,
                "longitude": TARGET_LONS,
            },
        )
        result.attrs = {"source": "GLORYS12V1", "regridded_to": "0.25deg"}
        result.to_netcdf(out_file)

        ds.close()
        result.close()
        del thetao_out, so_out
        gc.collect()
        elapsed = time.time() - t0
        logger.info("  Saved %s (%.1f min)", out_file.name, elapsed / 60)

    # Merge all years
    merged_file = output_dir / "glorys_025_merged.nc"
    if not merged_file.exists():
        logger.info("Merging all GLORYS years...")
        all_files = sorted(output_dir.glob("glorys_025_*.nc"))
        ds = xr.concat([xr.open_dataset(f) for f in all_files], dim="time")
        ds = ds.sel(time=slice(TRAIN_START, TRAIN_END))
        ds.to_netcdf(merged_file)
        for f in all_files:
            pass
        ds.close()
        gc.collect()
        logger.info("Merged GLORYS saved: %s", merged_file)


def process_sst(output_dir: Path) -> Path:
    """Process SST: fast regrid 0.05° → 0.25° in time chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sst_025.nc"
    if out_file.exists():
        logger.info("SST already processed, skipping")
        return out_file

    t0 = time.time()
    logger.info("Processing SST (regrid 0.05° -> 0.25° in chunks)...")
    ds = xr.open_dataset(SST_FILE)
    ds = ds.sel(time=slice(TRAIN_START, TRAIN_END))

    src_lats = ds.latitude.values
    src_lons = ds.longitude.values
    n_time = ds.sizes["time"]

    logger.info("  SST shape: %d x %d x %d -> %d x %d",
                n_time, ds.sizes["latitude"], ds.sizes["longitude"],
                len(TARGET_LATS), len(TARGET_LONS))

    tg_lats, tg_lons = np.meshgrid(TARGET_LATS, TARGET_LONS, indexing="ij")

    out_data = np.empty((n_time, len(TARGET_LATS), len(TARGET_LONS)), dtype=np.float32)

    time_chunk = 20
    for t_start in range(0, n_time, time_chunk):
        t_end = min(t_start + time_chunk, n_time)
        chunk = ds["analysed_sst"].isel(time=slice(t_start, t_end)).values
        for t_local in range(t_end - t_start):
            interp = RegularGridInterpolator(
                (src_lats, src_lons), chunk[t_local],
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            out_data[t_start + t_local] = interp((tg_lats, tg_lons))
        del chunk
        gc.collect()
        if t_start % 200 == 0:
            logger.info("    SST: %d/%d timesteps", t_end, n_time)

    result = xr.Dataset(
        {"sst": (["time", "latitude", "longitude"], out_data)},
        coords={"time": ds.time.values, "latitude": TARGET_LATS, "longitude": TARGET_LONS},
    )
    result.to_netcdf(out_file)
    ds.close()
    result.close()
    del out_data
    gc.collect()
    logger.info("SST saved in %.1f min", (time.time() - t0) / 60)
    return out_file


def _regrid_single_var(ds, var_name, src_lats, src_lons, n_time):
    """Regrid a single surface variable to target grid."""
    tg_lats, tg_lons = np.meshgrid(TARGET_LATS, TARGET_LONS, indexing="ij")
    data = ds[var_name].values
    out = np.empty((n_time, len(TARGET_LATS), len(TARGET_LONS)), dtype=np.float32)
    for t in range(0, n_time, 50):
        end = min(t + 50, n_time)
        for tt in range(t, end):
            interp = RegularGridInterpolator(
                (src_lats, src_lons), data[tt],
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            out[tt] = interp((tg_lats, tg_lons))
        if t % 500 == 0:
            logger.info("    %s: %d/%d", var_name, end, n_time)
    return out


def process_ssh(output_dir: Path) -> Path:
    """Process SSH: regrid to target 0.25° grid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "ssh_025.nc"
    if out_file.exists():
        logger.info("SSH already processed, skipping")
        return out_file

    logger.info("Processing SSH (regrid to 0.25°)...")
    ds = xr.open_dataset(SSH_FILE)
    ds = ds.sel(time=slice(TRAIN_START, TRAIN_END))
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    src_lats = ds[lat_name].values
    src_lons = ds[lon_name].values
    n_time = ds.sizes["time"]

    out = _regrid_single_var(ds, "sla", src_lats, src_lons, n_time)
    result = xr.Dataset(
        {"ssh": (["time", "latitude", "longitude"], out)},
        coords={"time": ds.time.values, "latitude": TARGET_LATS, "longitude": TARGET_LONS},
    )
    result.to_netcdf(out_file)
    ds.close(); result.close(); del out; gc.collect()
    logger.info("SSH saved: %s", out_file)
    return out_file


def process_sss(output_dir: Path) -> Path:
    """Process SSS: regrid to target 0.25° grid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sss_025.nc"
    if out_file.exists():
        logger.info("SSS already processed, skipping")
        return out_file

    logger.info("Processing SSS (regrid to 0.25°)...")
    ds = xr.open_dataset(SSS_FILE)
    ds = ds.sel(time=slice(TRAIN_START, TRAIN_END))
    if "depth" in ds.dims and ds.sizes["depth"] == 1:
        ds = ds.squeeze("depth", drop=True)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    src_lats = ds[lat_name].values
    src_lons = ds[lon_name].values
    n_time = ds.sizes["time"]

    out = _regrid_single_var(ds, "sos", src_lats, src_lons, n_time)
    result = xr.Dataset(
        {"sss": (["time", "latitude", "longitude"], out)},
        coords={"time": ds.time.values, "latitude": TARGET_LATS, "longitude": TARGET_LONS},
    )
    result.to_netcdf(out_file)
    ds.close(); result.close(); del out; gc.collect()
    logger.info("SSS saved: %s", out_file)
    return out_file


def process_wind(output_dir: Path) -> Path:
    """Process Wind: daily resample + regrid to target 0.25° grid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "wind_025.nc"
    if out_file.exists():
        logger.info("Wind already processed, skipping")
        return out_file

    t0 = time.time()
    logger.info("Processing Wind (daily resample + regrid)...")
    ds = xr.open_dataset(WIND_FILE)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    src_lats = ds[lat_name].values
    src_lons = ds[lon_name].values

    logger.info("  Resampling %d timesteps to daily...", ds.sizes["time"])
    ds = ds.resample(time="1D").mean()
    ds = ds.sel(time=slice(TRAIN_START, TRAIN_END))
    n_time = ds.sizes["time"]

    out_u = _regrid_single_var(ds, "eastward_wind", src_lats, src_lons, n_time)
    out_v = _regrid_single_var(ds, "northward_wind", src_lats, src_lons, n_time)

    result = xr.Dataset(
        {"wind_u": (["time", "latitude", "longitude"], out_u),
         "wind_v": (["time", "latitude", "longitude"], out_v)},
        coords={"time": ds.time.values, "latitude": TARGET_LATS, "longitude": TARGET_LONS},
    )
    result.to_netcdf(out_file)
    ds.close(); result.close(); del out_u, out_v; gc.collect()
    logger.info("Wind saved in %.1f min", (time.time() - t0) / 60)
    return out_file


def combine_surface_inputs(output_dir: Path) -> Path:
    """Combine all surface inputs into a single multi-channel dataset."""
    out_file = output_dir / "surface_combined.nc"
    if out_file.exists():
        logger.info("Surface inputs already combined, skipping")
        return out_file

    logger.info("Combining surface inputs...")
    sst = xr.open_dataset(output_dir / "sst_025.nc")
    ssh = xr.open_dataset(output_dir / "ssh_025.nc")
    sss = xr.open_dataset(output_dir / "sss_025.nc")
    wind = xr.open_dataset(output_dir / "wind_025.nc")

    common_times = np.intersect1d(sst.time.values, ssh.time.values)
    common_times = np.intersect1d(common_times, sss.time.values)
    common_times = np.intersect1d(common_times, wind.time.values)
    logger.info("  Common time range: %d days", len(common_times))

    sst = sst.sel(time=common_times)
    ssh = ssh.sel(time=common_times)
    sss = sss.sel(time=common_times)
    wind = wind.sel(time=common_times)

    combined = xr.merge([sst, ssh, sss, wind], join="inner")
    combined.to_netcdf(out_file)

    sst.close(); ssh.close(); sss.close(); wind.close(); combined.close()
    gc.collect()
    logger.info("Combined surface inputs: %s", out_file)
    return out_file


def split_and_save(surface_path: Path, glorys_path: Path, output_dir: Path) -> None:
    """Split data into train/val/test and save."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Splitting data into train/val/test...")

    surface = xr.open_dataset(surface_path)
    glorys = xr.open_dataset(glorys_path)
    target_var = "thetao"

    splits = {
        "train": (TRAIN_START, VAL_START),
        "val": (VAL_START, TEST_START),
        "test": (TEST_START, TRAIN_END),
    }

    for split_name, (start, end) in splits.items():
        s = surface.sel(time=slice(start, end))
        g = glorys.sel(time=slice(start, end))
        common = np.intersect1d(s.time.values, g.time.values)
        s = s.sel(time=common)
        g = g.sel(time=common)

        if len(common) == 0:
            logger.warning("  %s split has 0 timesteps, skipping", split_name)
            continue

        s.to_netcdf(output_dir / f"{split_name}_surface.nc")
        g[[target_var]].to_netcdf(output_dir / f"{split_name}_target.nc")
        logger.info("  %s: %d days", split_name, len(common))
        s.close(); g.close()

    surface.close(); glorys.close()
    gc.collect()

    # Compute normalization stats from training set
    logger.info("Computing normalization statistics from training set...")
    train_s = xr.open_dataset(output_dir / "train_surface.nc")
    train_t = xr.open_dataset(output_dir / "train_target.nc")

    stats = {}
    for var in train_s.data_vars:
        stats[var] = {"mean": float(train_s[var].mean()), "std": float(train_s[var].std())}
    stats[target_var] = {"mean": float(train_t[target_var].mean()), "std": float(train_t[target_var].std())}

    with open(output_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Normalization stats saved.")

    train_s.close(); train_t.close()


def run_harmonization_pipeline():
    """Run the full harmonization pipeline."""
    interim_dir = INTERIM_DIR / "harmonized"
    interim_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    logger.info("=== Data Harmonization Pipeline ===")
    logger.info("Domain: %.1f-%.1f N, %.1f-%.1f E", LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    logger.info("Target grid: %d x %d", len(TARGET_LATS), len(TARGET_LONS))

    logger.info("Step 1/6: Processing GLORYS (scipy regrid)...")
    process_glorys_yearly(interim_dir)

    logger.info("Step 2/6: Processing SST (scipy regrid)...")
    process_sst(interim_dir)

    logger.info("Step 3/6: Processing SSH (crop only)...")
    process_ssh(interim_dir)

    logger.info("Step 4/6: Processing SSS (crop only)...")
    process_sss(interim_dir)

    logger.info("Step 5/6: Processing Wind (daily resample + crop)...")
    process_wind(interim_dir)

    logger.info("Step 6/6: Combining surface inputs and splitting...")
    combine_surface_inputs(interim_dir)

    glorys_merged = interim_dir / "glorys_025_merged.nc"
    surface_combined = interim_dir / "surface_combined.nc"

    if glorys_merged.exists() and surface_combined.exists():
        split_and_save(surface_combined, glorys_merged, PROCESSED_DIR)
    else:
        logger.error("Missing merged files. Cannot split.")
        return

    logger.info("=== Harmonization Complete in %.1f min ===", (time.time() - t_total) / 60)
    logger.info("Processed files: %s", PROCESSED_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_harmonization_pipeline()
