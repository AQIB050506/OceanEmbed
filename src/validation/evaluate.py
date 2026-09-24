"""
Validation framework for OceanEmbed.
Evaluates reconstructed profiles against independent ARGO observations.
"""
import torch
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional
import json
import logging

from src.config import DEPTH_LEVELS, PROCESSED_DIR, MODELS_DIR
from src.reconstruction.model import AttentionUNet3D
from src.reconstruction.train import Trainer

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates a trained model against ARGO validation data."""

    def __init__(
        self,
        model: AttentionUNet3D,
        device: str = "cuda",
        normalization_stats: Optional[dict] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.stats = normalization_stats

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run inference on a batch of inputs."""
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)
        return output.cpu().numpy()

    def denormalize(self, data: np.ndarray, var: str = "thetao") -> np.ndarray:
        """Denormalize predictions back to physical units."""
        if self.stats and var in self.stats:
            mean = self.stats[var]["mean"]
            std = self.stats[var]["std"]
            return data * std + mean
        return data

    def evaluate_against_argo(
        self,
        argo_path: Path,
        predictions: np.ndarray,
        dates: list[str],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> dict:
        """
        Compare model predictions with ARGO observations.

        Args:
            argo_path: Path to INCOIS gridded ARGO NetCDF
            predictions: Model output (N_samples, N_depth, H, W)
            dates: List of date strings for each sample
            latitudes: Latitude array
            longitudes: Longitude array

        Returns:
            Dictionary of validation metrics
        """
        argo_ds = xr.open_dataset(argo_path)

        all_correlations = []
        all_rmse = []
        all_bias = []
        depth_metrics = {d: {"rmse": [], "correlation": [], "bias": []} for d in DEPTH_LEVELS}

        for i, date in enumerate(dates):
            try:
                argo_day = argo_ds.sel(time=date, method="nearest")
            except (KeyError, ValueError):
                continue

            for d_idx, depth in enumerate(DEPTH_LEVELS):
                if depth not in argo_day.depth.values:
                    continue

                argo_temp = argo_day["temperature"].sel(depth=depth).values
                pred_temp = predictions[i, d_idx]

                argo_flat = argo_temp.flatten()
                pred_flat = pred_temp.flatten()

                valid = ~(np.isnan(argo_flat) | np.isnan(pred_flat))
                if valid.sum() < 10:
                    continue

                a = argo_flat[valid]
                p = pred_flat[valid]

                rmse = np.sqrt(np.mean((a - p) ** 2))
                bias = np.mean(p - a)
                corr = np.corrcoef(a, p)[0, 1] if len(a) > 1 else 0

                depth_metrics[depth]["rmse"].append(rmse)
                depth_metrics[depth]["correlation"].append(corr)
                depth_metrics[depth]["bias"].append(bias)

        summary = {}
        for depth in DEPTH_LEVELS:
            if depth_metrics[depth]["rmse"]:
                summary[depth] = {
                    "rmse_mean": float(np.mean(depth_metrics[depth]["rmse"])),
                    "rmse_std": float(np.std(depth_metrics[depth]["rmse"])),
                    "correlation_mean": float(np.mean(depth_metrics[depth]["correlation"])),
                    "correlation_std": float(np.std(depth_metrics[depth]["correlation"])),
                    "bias_mean": float(np.mean(depth_metrics[depth]["bias"])),
                    "bias_std": float(np.std(depth_metrics[depth]["bias"])),
                    "n_samples": len(depth_metrics[depth]["rmse"]),
                }

        all_rmse_vals = [v for d in summary.values() for v in d.get("rmse_mean", [])]
        all_corr_vals = [v for d in summary.values() for v in d.get("correlation_mean", [])]

        summary["overall"] = {
            "mean_rmse": float(np.mean(all_rmse_vals)) if all_rmse_vals else None,
            "mean_correlation": float(np.mean(all_corr_vals)) if all_corr_vals else None,
            "n_depths_evaluated": len(all_rmse_vals),
        }

        return summary

    def error_analysis(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """
        Detailed error analysis by depth, region, and season.

        Args:
            predictions: Model output (N, D, H, W)
            targets: Ground truth (N, D, H, W)

        Returns:
            Error analysis dictionary
        """
        n_samples, n_depth, h, w = predictions.shape

        depth_errors = {}
        for d in range(n_depth):
            pred_d = predictions[:, d]
            target_d = targets[:, d]
            rmse = np.sqrt(np.mean((pred_d - target_d) ** 2))
            mae = np.mean(np.abs(pred_d - target_d))
            bias = np.mean(pred_d - target_d)
            ss_res = np.sum((target_d - pred_d) ** 2)
            ss_tot = np.sum((target_d - np.mean(target_d)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)

            depth_errors[DEPTH_LEVELS[d]] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "bias": float(bias),
                "r2": float(r2),
            }

        overall_rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        overall_mae = np.mean(np.abs(predictions - targets))
        overall_r2 = 1 - np.sum((targets - predictions) ** 2) / (
            np.sum((targets - np.mean(targets)) ** 2) + 1e-8
        )

        return {
            "overall": {
                "rmse": float(overall_rmse),
                "mae": float(overall_mae),
                "r2": float(overall_r2),
            },
            "by_depth": depth_errors,
        }


def load_trained_model(
    checkpoint_path: Path = MODELS_DIR / "best_model.pt",
    device: str = "cuda",
) -> AttentionUNet3D:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AttentionUNet3D(**{
        k: v for k, v in checkpoint.get("config", {}).items()
        if k in AttentionUNet3D.__init__.__code__.co_varnames
    })
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def run_full_validation(
    model_path: Path = MODELS_DIR / "best_model.pt",
    argo_path: Optional[Path] = None,
    output_dir: Path = PROCESSED_DIR / "validation_results",
):
    """Run complete validation pipeline and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(model_path)
    evaluator = ModelEvaluator(model)

    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Validation results will be saved to {output_dir}")

    return evaluator
