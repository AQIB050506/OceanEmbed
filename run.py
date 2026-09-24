#!/usr/bin/env python3
"""
OceanEmbed - Main Pipeline Runner
Run the full pipeline: preprocess → train → validate → demo
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, MODELS_DIR, PROCESSED_DIR


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_preprocess(args):
    """Run data preprocessing pipeline."""
    from src.preprocessing.harmonize import run_harmonization_pipeline
    logging.info("Starting preprocessing pipeline...")
    run_harmonization_pipeline()
    logging.info("Preprocessing complete.")


def run_train(args):
    """Run model training."""
    from src.preprocessing.dataset import create_dataloaders
    from src.reconstruction.model import build_model, count_parameters
    from src.reconstruction.train import Trainer

    import platform
    default_workers = 0 if platform.system() == "Windows" else 4

    logging.info("Building dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_surface=PROCESSED_DIR / "train_surface.nc",
        train_target=PROCESSED_DIR / "train_target.nc",
        val_surface=PROCESSED_DIR / "val_surface.nc" if (PROCESSED_DIR / "val_surface.nc").exists() else None,
        val_target=PROCESSED_DIR / "val_target.nc" if (PROCESSED_DIR / "val_target.nc").exists() else None,
        batch_size=args.batch_size,
        num_workers=default_workers,
        temporal_window=args.temporal_window,
    )

    logging.info(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        logging.info(f"Val samples: {len(val_loader.dataset)}")

    logging.info("Building model...")
    model = build_model(device=args.device)
    logging.info(f"Model parameters: {count_parameters(model):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr,
    )

    if args.resume:
        from src.config import MODELS_DIR
        trainer.load_checkpoint(MODELS_DIR / "final_model.pt")

    logging.info("Starting training...")
    trainer.train(n_epochs=args.epochs)
    logging.info("Training complete.")


def run_validate(args):
    """Run model validation."""
    from src.validation.evaluate import run_full_validation
    logging.info("Running validation...")
    evaluator = run_full_validation(
        model_path=MODELS_DIR / "best_model.pt",
        argo_path=PROCESSED_DIR / "argo_gridded.nc",
    )
    logging.info("Validation complete.")


def run_demo(args):
    """Start demo server."""
    import uvicorn
    from demo.backend.app import app
    logging.info(f"Starting demo server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed Pipeline Runner")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Preprocess
    p_pre = subparsers.add_parser("preprocess", help="Run preprocessing pipeline")
    p_pre.add_argument("--start-date", default="2010-01-01")
    p_pre.add_argument("--end-date", default="2025-12-31")

    # Train
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--temporal-window", type=int, default=5)
    p_train.add_argument("--device", default="cuda")
    p_train.add_argument("--resume", action="store_true")

    # Validate
    p_val = subparsers.add_parser("validate", help="Run validation")

    # Demo
    p_demo = subparsers.add_parser("demo", help="Start demo server")
    p_demo.add_argument("--host", default="0.0.0.0")
    p_demo.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.command == "preprocess":
        run_preprocess(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "validate":
        run_validate(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
