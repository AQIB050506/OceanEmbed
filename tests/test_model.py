"""Basic tests for OceanEmbed model architecture."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from src.config import MODEL_CONFIG


def test_model_forward_pass():
    """Test that model produces correct output shape."""
    from src.reconstruction.model import AttentionUNet3D, count_parameters

    model = AttentionUNet3D()
    params = count_parameters(model)
    assert params > 0, "Model should have parameters"

    batch = torch.randn(2, MODEL_CONFIG["input_channels"], 5, 101, 221)
    output = model(batch)

    assert output.shape == (2, MODEL_CONFIG["n_depth_levels"], 101, 221), \
        f"Expected shape (2, {MODEL_CONFIG['n_depth_levels']}, 101, 221), got {output.shape}"


def test_model_backward_pass():
    """Test that model gradients compute correctly."""
    from src.reconstruction.model import AttentionUNet3D

    model = AttentionUNet3D()
    batch = torch.randn(1, MODEL_CONFIG["input_channels"], 5, 101, 221)
    target = torch.randn(1, MODEL_CONFIG["n_depth_levels"], 101, 221)

    output = model(batch)
    loss = torch.nn.MSELoss()(output, target)
    loss.backward()

    has_grad = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    assert has_grad, "At least one parameter should have gradients"


def test_loss_function():
    """Test loss function."""
    from src.reconstruction.train import OceanEmbedLoss

    loss_fn = OceanEmbedLoss(n_depth_levels=MODEL_CONFIG["n_depth_levels"])
    pred = torch.randn(4, MODEL_CONFIG["n_depth_levels"], 32, 64)
    target = torch.randn(4, MODEL_CONFIG["n_depth_levels"], 32, 64)

    losses = loss_fn(pred, target)
    assert "total" in losses
    assert "mse" in losses
    assert losses["total"] > 0


def test_model_different_spatial_sizes():
    """Test model handles different spatial dimensions."""
    from src.reconstruction.model import AttentionUNet3D

    model = AttentionUNet3D()

    for h, w in [(32, 32), (64, 128), (101, 221)]:
        batch = torch.randn(1, MODEL_CONFIG["input_channels"], 5, h, w)
        output = model(batch)
        assert output.shape == (1, MODEL_CONFIG["n_depth_levels"], h, w)


def test_application_products():
    """Test application layer products."""
    from src.application.products import (
        compute_ocean_heat_content,
        detect_marine_heatwave,
        classify_cyclone_risk,
    )
    import numpy as np

    temp_last = np.random.uniform(5, 30, (50, 100, 15))
    ohc = compute_ocean_heat_content(temp_last)
    assert ohc.shape == (50, 100), f"OHC shape mismatch: {ohc.shape}"
    assert ohc.min() >= 0, "OHC should be non-negative"

    risk = classify_cyclone_risk(ohc)
    assert "risk_level" in risk
    assert "mean_ohc" in risk

    temp_first = np.random.uniform(5, 30, (15, 50, 100))
    mhw = detect_marine_heatwave(temp_first)
    assert "is_mhw" in mhw
    assert "mhw_fraction" in mhw


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
