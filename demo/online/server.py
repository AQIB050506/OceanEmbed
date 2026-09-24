"""
Online demo backend for OceanEmbed.
Fetches real-time satellite data, runs model inference, serves results.
"""
import torch
import numpy as np
import json
import time
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    MODELS_DIR, PROCESSED_DIR, DEPTH_LEVELS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION,
)
from src.reconstruction.model import AttentionUNet3D
from src.application.products import (
    compute_ocean_heat_content,
    detect_marine_heatwave,
    classify_cyclone_risk,
)
from realtime_fetcher import fetch_realtime_data

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RealtimeInferenceEngine:

    def __init__(self):
        self.model = None
        self.norm_stats = None

    def load_model(self):
        ckpt = torch.load(MODELS_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
        self.model = AttentionUNet3D()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.model.to(DEVICE)
        with open(PROCESSED_DIR / "normalization_stats.json") as f:
            self.norm_stats = json.load(f)
        logger.info("Model loaded (epoch %d)", ckpt.get("epoch", -1))

    @torch.no_grad()
    def predict(self, model_input):
        tensor = torch.from_numpy(model_input).float().unsqueeze(0).to(DEVICE)
        output = self.model(tensor)
        pred = output.cpu().numpy()[0]
        mean = self.norm_stats["thetao"]["mean"]
        std = self.norm_stats["thetao"]["std"]
        return pred * std + mean


def latlon_to_grid(lat, lon):
    lat_idx = int((lat - LAT_MIN) / RESOLUTION)
    lon_idx = int((lon - LON_MIN) / RESOLUTION)
    return max(0, min(lat_idx, 100)), max(0, min(lon_idx, 220))


def climatological_profile(sst, depths):
    """Mixed-layer + thermocline fallback for flat/unrealistic model outputs."""
    depths = np.array(depths, dtype=np.float32)
    t_deep = 4.0
    mld = 30.0
    scale = 80.0
    profile = t_deep + (sst - t_deep) * np.exp(-np.maximum(depths - mld, 0) / scale)
    profile[depths <= mld] = sst
    return profile


def is_flat_profile(temps):
    temps = np.asarray(temps)
    grads = np.diff(temps)
    sign_changes = np.sum(np.abs(np.diff(np.sign(grads))) > 1)
    noisy = sign_changes > 3
    unrealistic = np.std(temps) < 1.0 or (temps[0] > 25.0 and temps[-1] > 15.0)
    return unrealistic or noisy


def create_app():
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="OceanEmbed Real-Time Demo")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    engine = RealtimeInferenceEngine()
    cache = {"data": None, "preds": None, "timestamp": None}

    @app.on_event("startup")
    async def startup():
        engine.load_model()

    class ProfileQuery(BaseModel):
        latitude: float
        longitude: float

    @app.get("/api/realtime")
    async def get_realtime():
        now = time.time()
        if cache["data"] and cache["timestamp"] and (now - cache["timestamp"]) < 600:
            sst = cache["data"]["sst"]
            return {
                "date": cache["data"]["date"],
                "from_cache": True,
                "sst_mean": round(float(np.nanmean(sst)), 2),
                "message": "Cached data",
            }
        try:
            data = fetch_realtime_data()
            cache["data"] = data
            cache["timestamp"] = now
            t0 = time.time()
            preds = engine.predict(data["model_input"])
            cache["preds"] = preds
            return {
                "date": data["date"],
                "from_cache": False,
                "inference_time_s": round(time.time() - t0, 2),
                "sst_mean": round(float(np.nanmean(data["sst"])), 2),
                "message": "Fresh prediction",
            }
        except Exception as e:
            logger.error("Realtime failed: %s", e)
            return {"error": str(e)}

    @app.post("/api/profile")
    async def get_profile(query: ProfileQuery):
        if cache["preds"] is None:
            return {"error": "No predictions. Call /api/realtime first."}
        lat_idx, lon_idx = latlon_to_grid(query.latitude, query.longitude)
        temps = cache["preds"][:, lat_idx, lon_idx].tolist()
        source = "model"

        # Fallback if model output is flat/noisy at this point
        if is_flat_profile(temps):
            temps = climatological_profile(temps[0], DEPTH_LEVELS).tolist()
            source = "climatology_fallback"

        point_temp = np.array(temps)
        ohc = compute_ocean_heat_content(
            point_temp[np.newaxis, np.newaxis, :], DEPTH_LEVELS, (0, 200)
        )
        ohc_kj = float(ohc[0, 0] / 1e7)

        max_grad = 0
        thermo_idx = 3
        for i in range(1, len(temps)):
            grad = abs(temps[i] - temps[i - 1])
            if grad > max_grad:
                max_grad = grad
                thermo_idx = i

        mhw = detect_marine_heatwave(
            cache["preds"], DEPTH_LEVELS
        )

        ohc_full = compute_ocean_heat_content(
            np.transpose(cache["preds"], (1, 2, 0)), DEPTH_LEVELS, (0, 200)
        )
        cr = classify_cyclone_risk(ohc_full)

        return {
            "latitude": query.latitude,
            "longitude": query.longitude,
            "depths": DEPTH_LEVELS,
            "temperature": [round(t, 2) for t in temps],
            "source": source,
            "sst": round(temps[0], 2),
            "ohc_kj_cm2": round(ohc_kj, 1),
            "thermocline_depth_m": DEPTH_LEVELS[thermo_idx],
            "mhw": {
                "detected": mhw["is_mhw"],
                "fraction": round(mhw["mhw_fraction"], 4),
                "intensity": round(mhw["max_intensity"], 4),
            },
            "cyclone_risk": {
                "mean_ohc_kj": round(cr["mean_ohc"], 2),
                "pct_favorable": round(cr["pct_favorable"], 2),
                "pct_high_risk": round(cr["pct_high_risk"], 2),
            },
        }

    @app.get("/api/sst_map")
    async def get_sst_map():
        if cache["data"] is None:
            return {"error": "No data"}
        sst = cache["data"]["sst"]
        return {
            "date": cache["data"]["date"],
            "sst": [[round(v, 2) for v in row] for row in sst.tolist()],
            "lats": [round(LAT_MIN + i * RESOLUTION, 2) for i in range(sst.shape[0])],
            "lons": [round(LON_MIN + j * RESOLUTION, 2) for j in range(sst.shape[1])],
        }

    @app.get("/api/pred_map")
    async def get_pred_map(depth_idx: int = 0):
        if cache["preds"] is None:
            return {"error": "No predictions"}
        pred = cache["preds"][depth_idx]
        return {
            "depth": DEPTH_LEVELS[depth_idx],
            "temperature": [[round(v, 1) for v in row] for row in pred.tolist()],
        }

    @app.get("/api/volume")
    async def get_volume(stride: int = 2):
        """
        All 15 depth layers downsampled by `stride` for the 3D viewer.
        Returns flattened Float32-style arrays (rounded to 1 decimal) per depth.
        """
        if cache["preds"] is None:
            return {"error": "No predictions"}
        preds = cache["preds"]  # (15, 101, 221)
        layers = []
        for d in range(preds.shape[0]):
            layer = preds[d, ::stride, ::stride]
            layers.append([round(float(v), 1) for v in layer.ravel()])
        h = preds.shape[1] // stride + (1 if preds.shape[1] % stride else 0)
        w = preds.shape[2] // stride + (1 if preds.shape[2] % stride else 0)
        # sst input layer for the surface
        sst = cache["data"]["sst"][::stride, ::stride] if cache["data"] else None
        return {
            "date": cache["data"]["date"] if cache["data"] else None,
            "depths": DEPTH_LEVELS,
            "n_lat": preds.shape[1],
            "n_lon": preds.shape[2],
            "stride": stride,
            "grid_h": h,
            "grid_w": w,
            "layers": layers,
            "sst": [round(float(v), 1) for v in sst.ravel()] if sst is not None else None,
            "bounds": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX},
        }

    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
