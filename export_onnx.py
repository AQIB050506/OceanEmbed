"""Export best_model.pt to ONNX for CPU serverless deployment (Vercel).

Produces demo/online/assets/model.onnx (single input 'input', single output 'output').
Mirrors RealtimeInferenceEngine.predict: model(x) with climatology=None.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from src.reconstruction.model import AttentionUNet3D

CKPT = ROOT / "models" / "best_model.pt"
OUT = ROOT / "demo" / "online" / "assets" / "model.onnx"


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = AttentionUNet3D()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 5, 5, 101, 221)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(OUT),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )

    with torch.no_grad():
        ref = model(dummy).numpy()
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})[0]
    d = float(np.abs(ref - out).max())
    size_mb = OUT.stat().st_size / 1e6
    print(f"ONNX written: {OUT} ({size_mb:.1f} MB)  epoch={ckpt.get('epoch')}")
    print(f"parity on export dummy: max|torch-onnx| = {d:.3e}  {'PASS' if d < 1e-3 else 'FAIL'}")
    if d >= 1e-3:
        sys.exit(1)


if __name__ == "__main__":
    main()
