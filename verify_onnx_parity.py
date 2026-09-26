"""Gate: torch vs ONNX parity + CPU inference benchmark (Vercel = 1 vCPU).

Checks raw and denormalized outputs on a random input and a real fetched input.
Exits 1 if max|diff| >= 1e-3.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "demo" / "online"))

import numpy as np
import torch
import onnxruntime as ort
from src.reconstruction.model import AttentionUNet3D

CKPT = ROOT / "models" / "best_model.pt"
ONNX = ROOT / "demo" / "online" / "assets" / "model.onnx"
STATS = ROOT / "data" / "processed" / "normalization_stats.json"
TOL = 1e-3
RUNS = 5


def load_torch():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = AttentionUNet3D()
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def main():
    torch_model = load_torch()
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    with open(STATS) as f:
        stats = json.load(f)
    mean, std = stats["thetao"]["mean"], stats["thetao"]["std"]

    inputs = {"random N(0,1)": np.random.randn(1, 5, 5, 101, 221).astype(np.float32)}
    try:
        from realtime_fetcher import fetch_realtime_data
        real = fetch_realtime_data()["model_input"].astype(np.float32)[None]
        inputs["real fetched"] = real
        print("real input: fetched OK", real.shape)
    except Exception as e:
        print(f"real input: fetch failed ({e}); random only")

    worst_raw = 0.0
    worst_den = 0.0
    for name, x in inputs.items():
        with torch.no_grad():
            t = torch_model(torch.from_numpy(x)).numpy()
        o = sess.run(None, {"input": x})[0]
        d_raw = float(np.abs(t - o).max())
        d_den = float(np.abs((t - o) * std).max())
        worst_raw = max(worst_raw, d_raw)
        worst_den = max(worst_den, d_den)
        print(f"[{name}] max|d| raw={d_raw:.3e}  denormalized={d_den:.4f} K")

    print("\nCPU benchmark (ONNX Runtime, single session):")
    x = inputs["real fetched"] if "real fetched" in inputs else inputs["random N(0,1)"]
    sess.run(None, {"input": x})  # warmup
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append(time.perf_counter() - t0)
    times.sort()
    print(f"  n={RUNS}  median={times[len(times)//2]:.2f}s  min={times[0]:.2f}s  max={times[-1]:.2f}s")

    ok = worst_raw < TOL
    print(f"\nPARITY {'PASS' if ok else 'FAIL'} (tol {TOL}, worst raw {worst_raw:.3e}, denorm {worst_den:.4f} K)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
