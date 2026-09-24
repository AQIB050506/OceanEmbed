"""
Dataset and DataLoader for OceanEmbed training.
PyTorch Dataset that loads preprocessed NetCDF files and yields
(input_surface, target_temperature) pairs.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from src.config import (
    N_DEPTH_LEVELS, INPUT_VARIABLES,
    TRAINING_CONFIG, PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

# Order of surface input channels
INPUT_ORDER = ["sst", "ssh", "sss", "wind_u", "wind_v"]


class OceanTemperatureDataset(Dataset):
    """
    PyTorch Dataset for surface-to-subsurface temperature reconstruction.

    Loads harmonized surface inputs and GLORYS temperature targets,
    yielding (input_tensor, target_tensor) pairs for training.

    Input: (N_channels * temporal_window, H, W) surface variables
    Target: (N_depth, H, W) temperature profiles
    """

    def __init__(
        self,
        surface_path: Path,
        target_path: Path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        input_vars: list[str] = None,
        target_var: str = "thetao",
        temporal_window: int = 1,
        augment: bool = False,
        normalize: bool = True,
        stats: Optional[dict] = None,
    ):
        self.target_var = target_var
        self.temporal_window = temporal_window
        self.augment = augment
        self.normalize = normalize
        self.input_vars = input_vars or INPUT_ORDER

        self.surface_ds = xr.open_dataset(surface_path)
        self.target_ds = xr.open_dataset(target_path)

        if start_date:
            self.surface_ds = self.surface_ds.sel(time=slice(start_date, end_date))
            self.target_ds = self.target_ds.sel(time=slice(start_date, end_date))

        # Align time axes
        common_times = np.intersect1d(
            self.surface_ds.time.values, self.target_ds.time.values
        )
        self.surface_ds = self.surface_ds.sel(time=common_times)
        self.target_ds = self.target_ds.sel(time=common_times)
        self.dates = common_times

        self.stats = stats
        if self.normalize and self.stats is None:
            self.stats = self._compute_stats()

        self.n_lat = len(self.surface_ds.latitude)
        self.n_lon = len(self.surface_ds.longitude)

        logger.info(
            "Dataset loaded: %d timesteps, %d x %d spatial, "
            "temporal_window=%d, augment=%s",
            len(self.dates), self.n_lat, self.n_lon,
            self.temporal_window, self.augment,
        )

    def __del__(self):
        if hasattr(self, 'surface_ds') and self.surface_ds is not None:
            self.surface_ds.close()
        if hasattr(self, 'target_ds') and self.target_ds is not None:
            self.target_ds.close()

    def _compute_stats(self) -> dict:
        """Compute normalization statistics from the dataset."""
        stats = {}
        for var in self.input_vars:
            if var in self.surface_ds:
                data = self.surface_ds[var].values
                stats[var] = {
                    "mean": float(np.nanmean(data)),
                    "std": float(np.nanstd(data)),
                }
        if self.target_var in self.target_ds:
            target_data = self.target_ds[self.target_var].values
            stats[self.target_var] = {
                "mean": float(np.nanmean(target_data)),
                "std": float(np.nanstd(target_data)),
            }
        return stats

    def __len__(self) -> int:
        return max(0, len(self.dates) - self.temporal_window + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_indices = range(idx, idx + self.temporal_window)
        input_channels = []

        for var in self.input_vars:
            if var in self.surface_ds:
                var_data = []
                for t in time_indices:
                    arr = self.surface_ds[var].isel(time=t).values.astype(np.float32)
                    if self.normalize and var in self.stats:
                        arr = (arr - self.stats[var]["mean"]) / (
                            self.stats[var]["std"] + 1e-8
                        )
                    var_data.append(arr)
                # Stack temporal window as T dimension: (T, H, W)
                input_channels.append(np.stack(var_data, axis=0))

        # Stack variable channels: (C, T, H, W)
        input_tensor = np.stack(input_channels, axis=0)

        # Target is the last timestep's temperature profile: (N_depth, H, W)
        target_idx = idx + self.temporal_window - 1
        target = (
            self.target_ds[self.target_var]
            .isel(time=target_idx)
            .values.astype(np.float32)
        )
        if self.normalize and self.target_var in self.stats:
            target = (target - self.stats[self.target_var]["mean"]) / (
                self.stats[self.target_var]["std"] + 1e-8
            )

        # Replace NaN with 0
        target = np.nan_to_num(target, nan=0.0)
        input_tensor = np.nan_to_num(input_tensor, nan=0.0)

        if self.augment:
            input_tensor, target = self._augment(input_tensor, target)

        return (
            torch.from_numpy(input_tensor.copy()),
            torch.from_numpy(target.copy()),
        )

    def _augment(
        self, inp: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation (random flips). Input is (C, T, H, W), target is (D, H, W)."""
        if np.random.random() > 0.5:
            inp = inp[:, :, :, ::-1].copy()
            target = target[:, :, ::-1].copy()
        if np.random.random() > 0.5:
            inp = inp[:, :, ::-1, :].copy()
            target = target[:, ::-1, :].copy()
        return inp, target

    def get_date(self, idx: int) -> str:
        """Get the date string for a given index."""
        return str(self.dates[idx + self.temporal_window - 1])[:10]


def create_dataloaders(
    train_surface: Path,
    train_target: Path,
    val_surface: Optional[Path] = None,
    val_target: Optional[Path] = None,
    batch_size: int = None,
    num_workers: int = None,
    temporal_window: int = 26,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation DataLoaders."""
    batch_size = batch_size or TRAINING_CONFIG.get("batch_size", 8)
    num_workers = num_workers if num_workers is not None else TRAINING_CONFIG.get("num_workers", 4)

    logger.info("Creating dataloaders with temporal_window=%d, batch_size=%d",
                temporal_window, batch_size)

    train_ds = OceanTemperatureDataset(
        train_surface, train_target,
        temporal_window=temporal_window,
        augment=True,
    )

    val_ds = None
    val_loader = None
    if val_surface and val_target and Path(val_surface).exists() and Path(val_target).exists():
        val_ds = OceanTemperatureDataset(
            val_surface, val_target,
            temporal_window=temporal_window,
            augment=False,
            stats=train_ds.stats,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    logger.info("Train samples: %d | Val samples: %s",
                len(train_ds), len(val_ds) if val_ds else "N/A")

    return train_loader, val_loader
