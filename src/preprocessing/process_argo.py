"""
Process raw INCOIS Argo data from ERDDAP row format into gridded format.
Bins Argo observations into 0.25° grid cells and standard depth levels.

GPU-accelerated via PyTorch CUDA with chunked time processing.
Avoids allocating the full 44 GB dense grid — processes ~500 dates per
chunk on the RTX 4050 (6 GB VRAM), then saves incrementally.
"""
import gc
import numpy as np
import torch
import xarray as xr
from pathlib import Path
import logging

from src.config import (
    DEPTH_LEVELS, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    RESOLUTION, DATA_DIR, PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

RAW_ARGO_PATH = DATA_DIR / "incois_argo" / "Indian_ARGO_Floats_4298_d1fb_dfa5_U1789797672771.nc"
OUTPUT_PATH = PROCESSED_DIR / "argo_gridded.nc"

# ── Chunk size: dates per GPU batch ────────────────────────────────────────
# 100 dates × 15 depth × 101 lat × 221 lon × 4 bytes ≈ 134 MB per grid tensor
# With sum+count+mean: ~400 MB per chunk — safe for 6 GB VRAM
CHUNK_SIZE = 100
# ── Depth matching tolerance (meters) ──────────────────────────────────────
DEPTH_TOL = 10.0
# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-computed tensors (set once, reused across chunks)
_DEPTH_TENSORS = {}  # filled by _init_depth_tensors()


# ── Filtering functions (CPU / xarray — lightweight) ───────────────────────

def load_raw_argo(path: Path = RAW_ARGO_PATH) -> xr.Dataset:
    """Load raw ERDDAP Argo data."""
    logger.info("Loading raw Argo data from %s", path)
    ds = xr.open_dataset(path)
    logger.info("Raw data: %d rows, time %s to %s",
                len(ds.row),
                str(ds.time.min().values)[:10],
                str(ds.time.max().values)[:10])
    return ds


def filter_qc(ds: xr.Dataset) -> xr.Dataset:
    """Keep only good quality observations (QC = '1' or '2')."""
    n_before = len(ds.row)
    temp_qc = ds["TEMP_QC"].values.astype(str)
    pres_qc = ds["PRES_QC"].values.astype(str)
    good_mask = (
        ((temp_qc == "1") | (temp_qc == "2") | (temp_qc == " ")) &
        ((pres_qc == "1") | (pres_qc == "2") | (pres_qc == " "))
    )
    ds_filtered = ds.isel(row=good_mask)
    n_after = len(ds_filtered.row)
    logger.info("QC filter: %d -> %d rows (%.1f%% kept)",
                n_before, n_after, 100 * n_after / n_before)
    return ds_filtered


def filter_domain(ds: xr.Dataset) -> xr.Dataset:
    """Keep only observations within our domain."""
    n_before = len(ds.row)
    mask = (
        (ds.latitude >= LAT_MIN) &
        (ds.latitude <= LAT_MAX) &
        (ds.longitude >= LON_MIN) &
        (ds.longitude <= LON_MAX)
    )
    ds_filtered = ds.isel(row=mask)
    n_after = len(ds_filtered.row)
    logger.info("Domain filter (%.1f-%.1f N, %.1f-%.1f E): %d -> %d rows",
                LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, n_before, n_after)
    return ds_filtered


def filter_pressure(ds: xr.Dataset, max_pressure: float = 500.0) -> xr.Dataset:
    """Keep only observations above max_pressure (shallow enough for our depth levels)."""
    n_before = len(ds.row)
    mask = (ds.PRES >= 0) & (ds.PRES <= max_pressure)
    ds_filtered = ds.isel(row=mask)
    n_after = len(ds_filtered.row)
    logger.info("Pressure filter (0-%.0f dbar): %d -> %d rows",
                max_pressure, n_before, n_after)
    return ds_filtered


# ── GPU binning helpers ────────────────────────────────────────────────────

def _init_depth_tensors():
    """Pre-compute depth lookup tensors on GPU (called once)."""
    depth_levels = torch.tensor(DEPTH_LEVELS, dtype=torch.float32, device=DEVICE)
    _DEPTH_TENSORS["levels"] = depth_levels
    _DEPTH_TENSORS["n_levels"] = len(DEPTH_LEVELS)


def _bin_chunk_gpu(
    times_chunk: np.ndarray,
    lats_all: np.ndarray,
    lons_all: np.ndarray,
    pres_all: np.ndarray,
    temps_all: np.ndarray,
    time_indices_all: np.ndarray,
    n_lats: int,
    n_lons: int,
    n_depth: int,
) -> np.ndarray:
    """
    Bin observations for a chunk of dates onto the grid using PyTorch CUDA.

    Vectorised: no Python loops over observations.

    Parameters
    ----------
    times_chunk      : array of datetime64 values for this chunk
    lats_all         : filtered latitude array  (N,)
    lons_all         : filtered longitude array  (N,)
    pres_all         : filtered pressure array   (N,)
    temps_all        : filtered temperature array (N,)
    time_indices_all : integer index into times_chunk for each row  (N,)
    n_lats, n_lons, n_depth : grid dimensions

    Returns
    -------
    temp_grid : np.ndarray shape (n_dates_chunk, n_depth, n_lats, n_lons), float32
    """
    n_dates = len(times_chunk)

    # ── Move coordinate arrays to GPU ──────────────────────────────────────
    lats_gpu = torch.tensor(lats_all, dtype=torch.float32, device=DEVICE)
    lons_gpu = torch.tensor(lons_all, dtype=torch.float32, device=DEVICE)
    pres_gpu = torch.tensor(pres_all, dtype=torch.float32, device=DEVICE)
    temps_gpu = torch.tensor(temps_all, dtype=torch.float32, device=DEVICE)
    time_idx_gpu = torch.tensor(time_indices_all, dtype=torch.long, device=DEVICE)

    # ── Compute grid indices (vectorised) ───────────────────────────────────
    lat_idx = torch.round((lats_gpu - LAT_MIN) / RESOLUTION).long()
    lon_idx = torch.round((lons_gpu - LON_MIN) / RESOLUTION).long()
    depths = pres_gpu / 1.02  # pressure to depth

    # Closest depth level for every observation
    depth_levels = _DEPTH_TENSORS["levels"]  # (15,)
    depth_diffs = torch.abs(depths.unsqueeze(1) - depth_levels.unsqueeze(0))  # (N, 15)
    depth_idx = torch.argmin(depth_diffs, dim=1)  # (N,)
    closest_diff = depth_diffs[torch.arange(len(depth_idx), device=DEVICE), depth_idx]

    # ── Validity mask ───────────────────────────────────────────────────────
    valid = (
        (lat_idx >= 0) & (lat_idx < n_lats) &
        (lon_idx >= 0) & (lon_idx < n_lons) &
        (depth_idx >= 0) & (depth_idx < n_depth) &
        (~torch.isnan(temps_gpu)) &
        (closest_diff <= DEPTH_TOL)
    )

    # Filter to valid observations only
    v_time_idx = time_idx_gpu[valid]
    v_depth_idx = depth_idx[valid]
    v_lat_idx = lat_idx[valid]
    v_lon_idx = lon_idx[valid]
    v_temps = temps_gpu[valid]

    logger.info("  GPU: %d / %d observations valid after index check",
                valid.sum().item(), len(valid))

    # ── Scatter-add into grid ───────────────────────────────────────────────
    flat_idx = (
        v_time_idx * (n_depth * n_lats * n_lons) +
        v_depth_idx * (n_lats * n_lons) +
        v_lat_idx * n_lons +
        v_lon_idx
    )

    n_elements = n_dates * n_depth * n_lats * n_lons

    # Use float32 for sum (temps are ~0-30°C, sufficient precision)
    sum_grid = torch.zeros(n_elements, dtype=torch.float32, device=DEVICE)
    count_grid = torch.zeros(n_elements, dtype=torch.float32, device=DEVICE)

    sum_grid.index_add_(0, flat_idx, v_temps)
    count_grid.index_add_(0, flat_idx, torch.ones_like(v_temps))

    # In-place divide to avoid allocating a third large tensor
    mask = count_grid > 0
    sum_grid[mask] /= count_grid[mask]

    # Reshape and mark empty cells as NaN
    temp_grid = sum_grid.reshape(n_dates, n_depth, n_lats, n_lons)
    temp_grid[~mask.reshape(n_dates, n_depth, n_lats, n_lons)] = float("nan")

    # Free GPU memory
    del lats_gpu, lons_gpu, pres_gpu, temps_gpu, time_idx_gpu
    del lat_idx, lon_idx, depths, depth_diffs, depth_idx, closest_diff
    del valid, v_time_idx, v_depth_idx, v_lat_idx, v_lon_idx, v_temps
    del flat_idx, sum_grid, count_grid, mask
    torch.cuda.empty_cache()

    return temp_grid.cpu().numpy()


# ── Main pipeline ──────────────────────────────────────────────────────────

def bin_to_grid(ds: xr.Dataset) -> xr.Dataset:
    """
    Bin Argo observations into 0.25° grid cells and standard depth levels.
    Processes in time chunks on GPU to avoid allocating the full 44 GB grid.
    """
    _init_depth_tensors()

    lats = np.arange(LAT_MIN, LAT_MAX + RESOLUTION, RESOLUTION)
    lons = np.arange(LON_MIN, LON_MAX + RESOLUTION, RESOLUTION)
    n_lats, n_lons = len(lats), len(lons)
    n_depth = len(DEPTH_LEVELS)

    times = np.unique(ds.time.values)
    n_times = len(times)
    logger.info("Binning %d observations into %d dates, %dx%d grid, %d depth levels",
                len(ds.row), n_times, n_lats, n_lons, n_depth)

    # ── Pre-extract all filtered arrays ONCE (CPU → GPU transfer per chunk) ─
    logger.info("Extracting filtered arrays to CPU...")
    lats_all = ds.latitude.values.astype(np.float32)
    lons_all = ds.longitude.values.astype(np.float32)
    pres_all = ds.PRES.values.astype(np.float32)
    temps_all = ds.TEMP.values.astype(np.float32)
    times_all = ds.time.values  # datetime64

    # Map every observation to its time-chunk index (integer)
    # time_to_idx: datetime64 -> index into times array
    time_to_idx = {t: i for i, t in enumerate(times)}
    time_indices = np.array([time_to_idx[t] for t in times_all], dtype=np.int64)

    del ds  # free xarray memory
    gc.collect()

    # ── Process in chunks ───────────────────────────────────────────────────
    n_chunks = (n_times + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_valid = 0
    total_cells = n_times * n_depth * n_lats * n_lons

    # ── Write directly to NetCDF in append mode to avoid 38 GB concatenation ─
    first_write = True
    for chunk_i in range(n_chunks):
        start = chunk_i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_times)
        times_chunk = times[start:end]

        logger.info("Chunk %d/%d: dates %d-%d (%d dates)",
                     chunk_i + 1, n_chunks, start, end - 1, len(times_chunk))

        chunk_mask = (time_indices >= start) & (time_indices < end)
        chunk_time_indices = time_indices[chunk_mask] - start

        if chunk_mask.sum() == 0:
            logger.info("  No observations in this chunk, skipping.")
            chunk_grid = np.full((len(times_chunk), n_depth, n_lats, n_lons),
                                 np.nan, dtype=np.float32)
        else:
            chunk_grid = _bin_chunk_gpu(
                times_chunk=times_chunk,
                lats_all=lats_all[chunk_mask],
                lons_all=lons_all[chunk_mask],
                pres_all=pres_all[chunk_mask],
                temps_all=temps_all[chunk_mask],
                time_indices_all=chunk_time_indices,
                n_lats=n_lats,
                n_lons=n_lons,
                n_depth=n_depth,
            )

        chunk_valid = int(np.sum(~np.isnan(chunk_grid)))
        total_valid += chunk_valid

        # Write this chunk directly to NetCDF (append after first write)
        chunk_ds = xr.Dataset(
            {
                "temperature": (["time", "depth", "latitude", "longitude"], chunk_grid),
            },
            coords={
                "time": times_chunk,
                "depth": DEPTH_LEVELS,
                "latitude": lats,
                "longitude": lons,
            },
        )

        if first_write:
            chunk_ds.to_netcdf(OUTPUT_PATH, unlimited_dims=["time"])
            first_write = False
        else:
            # Append by reading existing + new, but only for the time dimension
            # Use netCDF4 directly to append without loading full file
            import netCDF4 as nc4
            with nc4.Dataset(OUTPUT_PATH, "a") as ncf:
                t_start = ncf.dimensions["time"].size
                for t_local in range(len(times_chunk)):
                    ncf.variables["temperature"][t_start + t_local] = chunk_grid[t_local]

        del chunk_grid, chunk_ds
        gc.collect()

        if (chunk_i + 1) % 50 == 0:
            logger.info("  Progress: %d/%d chunks, %d valid cells so far",
                        chunk_i + 1, n_chunks, total_valid)

    logger.info("Gridding complete: %d/%d cells valid (%.2f%%)",
                total_valid, total_cells, 100 * total_valid / total_cells)


def main():
    """Full pipeline: load -> filter -> grid -> save."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    logger.info("Using device: %s", DEVICE)
    if DEVICE.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info("VRAM: %.1f GB total", torch.cuda.get_device_properties(0).total_memory / 1e9)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove old output if exists
    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        logger.info("Removed old output file")

    ds = load_raw_argo()
    ds = filter_qc(ds)
    ds = filter_domain(ds)
    ds = filter_pressure(ds)

    bin_to_grid(ds)

    logger.info("Saved gridded Argo to %s", OUTPUT_PATH)
    logger.info("Output file size: %.1f MB", OUTPUT_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
