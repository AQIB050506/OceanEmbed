#!/usr/bin/env python3
"""Lightweight training script for OceanEmbed — laptop-friendly."""
import sys
import time
import logging
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from src.config import PROCESSED_DIR, MODELS_DIR, MODEL_CONFIG
from src.preprocessing.dataset import create_dataloaders
from src.reconstruction.model import AttentionUNet3D, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BATCH_SIZE = 2
EPOCHS = 3
LR = 1e-4
TEMPORAL_WINDOW = 5
DEVICE = "cuda"

def main():
    log.info("Device: %s", torch.cuda.get_device_name(0))
    log.info("VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    log.info("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_surface=PROCESSED_DIR / "train_surface.nc",
        train_target=PROCESSED_DIR / "train_target.nc",
        val_surface=PROCESSED_DIR / "val_surface.nc",
        val_target=PROCESSED_DIR / "val_target.nc",
        batch_size=BATCH_SIZE,
        num_workers=0,
        temporal_window=TEMPORAL_WINDOW,
    )
    log.info("Train samples: %d | Val samples: %d", len(train_loader.dataset), len(val_loader.dataset))

    model = AttentionUNet3D().to(DEVICE)
    log.info("Model params: %s", f"{count_parameters(model):,}")

    criterion = __import__("src.reconstruction.train", fromlist=["OceanEmbedLoss"]).OceanEmbedLoss(
        n_depth_levels=MODEL_CONFIG["n_depth_levels"]
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batch = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            losses["_total_tensor"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += losses["total"]
            n_batch += 1
            if i % 100 == 0:
                log.info("  [train] batch %d/%d  loss=%.4f", i, len(train_loader), losses["total"])

        avg_train = train_loss / max(n_batch, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        vn = 0
        val_mse = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                losses = criterion(outputs, targets)
                val_loss += losses["total"]
                val_mse += losses["mse"]
                vn += 1

        avg_val = val_loss / max(vn, 1)
        avg_mse = val_mse / max(vn, 1)
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        log.info(
            "Epoch %d/%d | Train: %.6f | Val: %.6f | MSE: %.6f | LR: %.2e | %.0fs",
            epoch + 1, EPOCHS, avg_train, avg_val, avg_mse, lr, elapsed,
        )

        if avg_val < best_val:
            best_val = avg_val
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val,
                "config": MODEL_CONFIG,
            }, MODELS_DIR / "best_model.pt")
            log.info("  -> Best model saved (val=%.6f)", best_val)

    log.info("Training complete. Best val loss: %.6f", best_val)


if __name__ == "__main__":
    main()
