"""
Training pipeline for OceanEmbed reconstruction model.
Includes loss functions, training loop, and experiment logging.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
import json
import logging
from datetime import datetime

from src.config import MODEL_CONFIG, TRAINING_CONFIG, MODELS_DIR
from src.reconstruction.model import AttentionUNet3D, build_model, count_parameters

logger = logging.getLogger(__name__)


# ─── Loss Functions ──────────────────────────────────────────────────────────

class UncertaintyAwareMSELoss(nn.Module):
    """
    Uncertainty-aware MSE loss from TS-Cast.
    Learns depth-dependent uncertainty weights.
    """

    def __init__(self, n_depth_levels: int = 15):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_depth_levels))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_vars = self.log_vars.view(1, -1, 1, 1) if self.log_vars.ndim == 1 else self.log_vars
        precision = torch.exp(-log_vars)
        squared_error = (pred - target) ** 2
        loss = 0.5 * (precision * squared_error + log_vars)
        return loss.mean()


class DensityConstraintLoss(nn.Module):
    """
    Physical constraint: penalize density inversions.
    Computes density from predicted T/S and checks for water column stability.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def eos_approx(t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Simplified equation of state for seawater density."""
        rho_0 = 1025.0
        alpha = 2.0e-4
        beta = 7.5e-4
        rho = rho_0 * (1 - alpha * (t - 20) + beta * (s - 35))
        return rho

    def forward(
        self, pred_t: torch.Tensor, pred_s: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if pred_s is None:
            return torch.tensor(0.0, device=pred_t.device)

        rho = self.eos_approx(pred_t, pred_s)
        drho_dz = rho[:, :, 1:, :, :] - rho[:, :, :-1, :, :]
        violation = F.relu(-drho_dz)
        return violation.mean()


class OceanEmbedLoss(nn.Module):
    """
    Combined loss for OceanEmbed training:
    1. Uncertainty-aware MSE
    2. Density constraint (optional)
    3. Depth-weighted MSE (emphasize thermocline)
    """

    def __init__(
        self,
        n_depth_levels: int = 15,
        use_density_constraint: bool = False,
        thermocline_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.uncertainty_loss = UncertaintyAwareMSELoss(n_depth_levels)
        self.density_loss = DensityConstraintLoss() if use_density_constraint else None
        self.use_density = use_density_constraint

        if thermocline_weights is None:
            weights = torch.ones(n_depth_levels)
            weights[5:12] = 2.0
            weights = weights / weights.sum() * n_depth_levels
        else:
            weights = thermocline_weights
        self.register_buffer("depth_weights", weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_s: Optional[torch.Tensor] = None,
    ) -> dict:
        B, D, H, W = pred.shape

        pred_weighted = pred * self.depth_weights.view(1, D, 1, 1)
        target_weighted = target * self.depth_weights.view(1, D, 1, 1)

        mse = F.mse_loss(pred_weighted, target_weighted)
        unc_loss = self.uncertainty_loss(pred, target)

        total_loss = mse + 0.1 * unc_loss
        losses = {"mse": mse.item(), "uncertainty": unc_loss.item(), "_total_tensor": total_loss}

        if self.use_density and pred_s is not None:
            d_loss = self.density_loss(pred, pred_s)
            total_loss = total_loss + 0.05 * d_loss
            losses["density"] = d_loss.item()

        losses["total"] = total_loss.item()
        return losses


# ─── Experiment Logger ────────────────────────────────────────────────────────

class ExperimentLogger:
    """Logs training experiments to JSON file."""

    def __init__(self, log_dir: Path = MODELS_DIR / "logs"):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"run_{self.run_id}.json"
        self.entries = []

    def log(self, entry: dict):
        entry["timestamp"] = datetime.now().isoformat()
        self.entries.append(entry)
        with open(self.log_file, "w") as f:
            json.dump(self.entries, f, indent=2, default=str)

    def log_config(self, config: dict):
        self.log({"type": "config", **config})

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, lr: float):
        self.log({
            "type": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        })

    def log_metrics(self, metrics: dict, phase: str = "validation"):
        self.log({"type": "metrics", "phase": phase, **metrics})


# ─── Training Loop ───────────────────────────────────────────────────────────

class Trainer:
    """Handles model training, validation, and checkpointing."""

    def __init__(
        self,
        model: AttentionUNet3D,
        train_loader,
        val_loader=None,
        device: str = "cuda",
        learning_rate: float = None,
        weight_decay: float = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        lr = learning_rate or MODEL_CONFIG["learning_rate"]
        wd = weight_decay or MODEL_CONFIG["weight_decay"]

        self.criterion = OceanEmbedLoss(n_depth_levels=MODEL_CONFIG["n_depth_levels"]).to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=MODEL_CONFIG["epochs"], eta_min=1e-6
        )

        self.logger = ExperimentLogger()
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.current_epoch = 0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

            losses["_total_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), TRAINING_CONFIG["gradient_clip_norm"]
            )
            self.optimizer.step()

            total_loss += losses["total"]
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> tuple[float, dict]:
        """Validate the model."""
        if self.val_loader is None:
            return float("inf"), {}

        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

            total_loss += losses["total"]
            n_batches += 1

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = self._compute_metrics(all_preds, all_targets)
        return avg_loss, metrics

    @staticmethod
    def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
        """Compute validation metrics."""
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        n_depth = preds.shape[1]
        depth_rmse = []
        for d in range(n_depth):
            d_rmse = np.sqrt(np.mean((preds[:, d] - targets[:, d]) ** 2))
            depth_rmse.append(float(d_rmse))

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "depth_rmse": depth_rmse,
        }

    def save_checkpoint(self, filepath: Path, metrics: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": MODEL_CONFIG,
            "metrics": metrics,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Checkpoint loaded: {filepath} (epoch {self.current_epoch})")

    def train(self, n_epochs: int = None):
        """Full training loop."""
        n_epochs = n_epochs or MODEL_CONFIG["epochs"]
        patience = TRAINING_CONFIG["early_stopping_patience"]

        self.logger.log_config({
            "model_params": count_parameters(self.model),
            "model_config": MODEL_CONFIG,
        })

        logger.info(f"Starting training for {n_epochs} epochs")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")

        for epoch in range(self.current_epoch, n_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log_epoch(epoch, train_loss, val_loss, lr)

            logger.info(
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {lr:.2e}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(
                    MODELS_DIR / "best_model.pt", metrics
                )
                logger.info(f"  New best model! Val RMSE: {metrics.get('rmse', 'N/A')}")
            else:
                self.patience_counter += 1

            if (epoch + 1) % TRAINING_CONFIG["checkpoint_interval"] == 0:
                self.save_checkpoint(MODELS_DIR / f"checkpoint_epoch_{epoch+1}.pt")

            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.save_checkpoint(MODELS_DIR / "final_model.pt")
        logger.info("Training complete")
