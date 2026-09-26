"""Precompute small deploy assets for Vercel (data/ and models/ are gitignored).

Outputs demo/online/assets/:
  norm_stats.json   copy of training normalization stats
  land_mask.npy     101x221 float32 (1=land) from test_target.nc NaN pattern
  argo_points.csv   cached recent in-domain Argo index subset
  model.onnx        produced separately by export_onnx.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

ASSETS = ROOT / "demo" / "online" / "assets"


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    src_stats = ROOT / "data" / "processed" / "normalization_stats.json"
    shutil.copy(src_stats, ASSETS / "norm_stats.json")
    print(f"norm_stats.json copied ({(ASSETS / 'norm_stats.json').stat().st_size} B)")

    import xarray as xr
    with xr.open_dataset(ROOT / "data" / "processed" / "test_target.nc") as ds:
        t = ds["thetao"].isel(time=0).values
    mask = np.isnan(t).any(axis=0)
    mask = np.nan_to_num(mask.astype(np.float32))
    assert mask.shape == (101, 221), mask.shape
    np.save(ASSETS / "land_mask.npy", mask)
    print(f"land_mask.npy written ({(ASSETS / 'land_mask.npy').stat().st_size} B, {100*mask.mean():.1f}% land)")

    src_argo = ROOT / "demo" / "online" / "cache" / "argo_points.csv"
    shutil.copy(src_argo, ASSETS / "argo_points.csv")
    n = sum(1 for _ in open(ASSETS / "argo_points.csv")) - 1
    print(f"argo_points.csv copied ({n} rows)")

    onnx = ASSETS / "model.onnx"
    print(f"model.onnx {'present' if onnx.exists() else 'MISSING (run python export_onnx.py)'} "
          f"({onnx.stat().st_size/1e6:.1f} MB)" if onnx.exists() else "model.onnx MISSING (run python export_onnx.py)")


if __name__ == "__main__":
    main()
