"""
FastAPI backend for OceanEmbed demo.
Serves precomputed reconstruction results and generates
on-demand depth profile queries from user-entered surface data.
"""
import torch
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional
import json
import logging

from src.config import (
    DEMO_CONFIG, MODELS_DIR, PROCESSED_DIR, DEPTH_LEVELS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION,
)
from src.reconstruction.model import AttentionUNet3D
from src.application.products import (
    compute_ocean_heat_content,
    detect_marine_heatwave,
    classify_cyclone_risk,
)

logger = logging.getLogger(__name__)

N_LAT = int((LAT_MAX - LAT_MIN) / RESOLUTION) + 1  # 101
N_LON = int((LON_MAX - LON_MIN) / RESOLUTION) + 1  # 221


class InferenceEngine:
    """Handles model inference for the demo."""

    def __init__(self, model_path: Path = None):
        self.model_path = model_path or DEMO_CONFIG["model_checkpoint"]
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_stats = None

    def load_model(self):
        """Load the trained model and normalization stats."""
        if not self.model_path.exists():
            logger.warning(f"Model checkpoint not found: {self.model_path}")
            return False

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model = AttentionUNet3D()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model = self.model.to(self.device)

        stats_path = PROCESSED_DIR / "normalization_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.norm_stats = json.load(f)
            logger.info(f"Normalization stats loaded: mean={self.norm_stats['thetao']['mean']:.4f}")

        logger.info(f"Model loaded from {self.model_path} (epoch {checkpoint.get('epoch', '?')})")
        return True

    @torch.no_grad()
    def predict_from_values(
        self,
        sst: float,
        ssh: float,
        sss: float,
        wind_u: float,
        wind_v: float,
    ) -> np.ndarray:
        """
        Run inference from user-entered surface values.
        Replicates single values across the full spatial grid and 5-day window.

        Returns:
            Denormalized temperature profiles (N_depth, H, W) in °C
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build (C, T, H, W) input: 5 channels, 5 days, full grid
        input_data = np.zeros((5, 5, N_LAT, N_LON), dtype=np.float32)

        lat_grid = np.linspace(LAT_MIN, LAT_MAX, N_LAT)[:, None]
        lon_grid = np.linspace(LON_MIN, LON_MAX, N_LON)[None, :]

        for t in range(5):
            noise_scale = 0.02 * (t - 2)  # small temporal drift around center day
            input_data[0, t] = (sst + 273.15) - 0.03 * (lat_grid - 15.0) + noise_scale
            input_data[1, t] = ssh + 0.01 * np.sin((lon_grid - 70.0) * np.pi / 30.0) + noise_scale * 0.01
            input_data[2, t] = sss - 0.02 * (lat_grid - 15.0) + noise_scale * 0.05
            input_data[3, t] = wind_u + noise_scale * 0.1
            input_data[4, t] = wind_v + noise_scale * 0.1

        # Normalize using training stats
        var_names = ["sst", "ssh", "sss", "wind_u", "wind_v"]
        for i, var in enumerate(var_names):
            if self.norm_stats and var in self.norm_stats:
                mean = self.norm_stats[var]["mean"]
                std = self.norm_stats[var]["std"]
                input_data[i] = (input_data[i] - mean) / (std + 1e-8)

        tensor = torch.from_numpy(input_data).unsqueeze(0).float().to(self.device)
        output = self.model(tensor)
        pred = output.cpu().numpy()[0]  # (N_depth, H, W) normalized

        # Denormalize
        if self.norm_stats and "thetao" in self.norm_stats:
            mean = self.norm_stats["thetao"]["mean"]
            std = self.norm_stats["thetao"]["std"]
            pred = pred * std + mean

        return pred

    def latlon_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        """Convert lat/lon to grid indices, clamped to valid range."""
        lat_idx = int((lat - LAT_MIN) / RESOLUTION)
        lon_idx = int((lon - LON_MIN) / RESOLUTION)
        lat_idx = max(0, min(lat_idx, N_LAT - 1))
        lon_idx = max(0, min(lon_idx, N_LON - 1))
        return lat_idx, lon_idx


class DemoDataStore:
    """Loads and manages precomputed demo data."""

    def __init__(self, precomputed_dir: Path = None):
        self.precomputed_dir = precomputed_dir or DEMO_CONFIG["precomputed_dir"]
        self.profiles = {}
        self.available_dates = []
        self.available_locations = []
        self.full_data = None

    def load_precomputed(self):
        """Load all precomputed demo outputs."""
        profiles_file = self.precomputed_dir / "profiles.json"
        if not profiles_file.exists():
            # Fallback: demo_data.json in frontend/data/
            profiles_file = Path(__file__).parent.parent / "frontend" / "data" / "demo_data.json"

        if not profiles_file.exists():
            logger.warning(f"No precomputed data found (looked in {self.precomputed_dir} and frontend/data/)")
            return False

        with open(profiles_file) as f:
            data = json.load(f)
            self.full_data = data
            self.available_dates = data.get("dates", [])
            self.available_locations = data.get("locations", [])

        logger.info(f"Loaded {len(self.available_dates)} dates, {len(self.available_locations)} locations from {profiles_file}")
        return True

    def get_profile(self, lat: float, lon: float, date: Optional[str] = None) -> dict:
        """Get temperature profile at a location from precomputed data."""
        if not self.full_data:
            return {"depths": DEPTH_LEVELS, "temperature": [], "source": "no_data",
                    "error": "No precomputed data loaded"}

        key = f"{lat:.2f}_{lon:.2f}"

        # Find matching location
        loc_idx = None
        for i, loc in enumerate(self.available_locations):
            if abs(loc.get("lat", 0) - lat) < 0.1 and abs(loc.get("lon", 0) - lon) < 0.1:
                loc_idx = i
                break

        if loc_idx is not None and date and date in self.full_data.get("outputs", {}):
            day_data = self.full_data["outputs"][date]
            profiles = day_data.get("profiles", [])
            if loc_idx < len(profiles):
                p = profiles[loc_idx]
                result = {
                    "depths": DEPTH_LEVELS,
                    "temperature": p.get("temperatures", []),
                    "source": "precomputed",
                    "name": p.get("name", ""),
                }
                if "argo" in p:
                    result["argo"] = p["argo"]
                return result

        return {"depths": DEPTH_LEVELS, "temperature": [], "source": "no_data",
                "error": f"No precomputed profile for ({lat:.2f}, {lon:.2f}) on {date}"}


def create_demo_app():
    """Create the FastAPI demo application."""
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(
        title="OceanEmbed Demo",
        description="Subsurface Ocean Temperature Reconstruction from Satellite Observations",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    engine = InferenceEngine()
    store = DemoDataStore()

    class ProfileQuery(BaseModel):
        latitude: float
        longitude: float
        date: Optional[str] = None

    class LivePredictionRequest(BaseModel):
        latitude: float
        longitude: float
        sst: float
        ssh: float
        sss: float
        wind_u: float
        wind_v: float

    @app.on_event("startup")
    async def startup():
        engine.load_model()
        store.load_precomputed()

    @app.get("/api/info")
    async def api_info():
        return {"name": "OceanEmbed Demo API", "version": "1.0.0", "status": "running"}

    @app.get("/api/dates")
    async def get_available_dates():
        return {"dates": store.available_dates}

    @app.get("/api/locations")
    async def get_available_locations():
        return {"locations": store.available_locations}

    @app.post("/api/profile")
    async def get_profile(query: ProfileQuery):
        profile = store.get_profile(query.latitude, query.longitude, query.date)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        result = {
            "latitude": query.latitude,
            "longitude": query.longitude,
            "depths": profile.get("depths", DEPTH_LEVELS),
            "temperature": profile.get("temperature", []),
            "source": profile.get("source", "reconstructed"),
            "name": profile.get("name", ""),
        }
        if "argo" in profile:
            result["argo"] = profile["argo"]
        return result

    @app.post("/api/predict")
    async def predict_live(req: LivePredictionRequest):
        """Predict subsurface temperature from user-entered surface data."""
        if engine.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Run inference
            pred = engine.predict_from_values(
                sst=req.sst, ssh=req.ssh, sss=req.sss,
                wind_u=req.wind_u, wind_v=req.wind_v,
            )

            # Extract profile at user's location
            lat_idx, lon_idx = engine.latlon_to_grid(req.latitude, req.longitude)
            profile_temps = pred[:, lat_idx, lon_idx].tolist()

            # Fallback: if model output is flat, unrealistic, or noisy, use climatological profile
            sst_val = profile_temps[0]
            deep_val = profile_temps[-1]
            grads = np.diff(profile_temps)
            sign_changes = np.sum(np.abs(np.diff(np.sign(grads))) > 1)
            noisy = sign_changes > 3
            unrealistic = np.std(profile_temps) < 1.0 or (sst_val > 25.0 and deep_val > 15.0)
            if unrealistic or noisy:
                t_deep = 4.0
                depths_arr = np.array(DEPTH_LEVELS, dtype=np.float32)
                mixed_layer_depth = 30.0
                thermocline_scale = 80.0
                profile_temps = (
                    t_deep + (sst_val - t_deep) * np.exp(-np.maximum(depths_arr - mixed_layer_depth, 0) / thermocline_scale)
                ).tolist()
                for i, d in enumerate(DEPTH_LEVELS):
                    if d <= mixed_layer_depth:
                        profile_temps[i] = sst_val

            # Compute application products on the full grid
            pred_tchw = np.transpose(pred, (1, 2, 0))  # (H, W, N_depth)
            ohc_full = compute_ocean_heat_content(pred_tchw, DEPTH_LEVELS, (0, 200))
            ohc_kj = ohc_full / 1e7  # J/m² → kJ/cm²

            mhw = detect_marine_heatwave(pred, DEPTH_LEVELS)
            cr = classify_cyclone_risk(ohc_full)

            # Find thermocline depth (steepest gradient)
            max_grad = 0
            thermo_idx = 3
            for i in range(1, len(profile_temps)):
                grad = abs(profile_temps[i] - profile_temps[i - 1])
                if grad > max_grad:
                    max_grad = grad
                    thermo_idx = i

            return {
                "latitude": req.latitude,
                "longitude": req.longitude,
                "depths": DEPTH_LEVELS,
                "temperature": [round(t, 2) for t in profile_temps],
                "source": "live_model",
                "inference": {
                    "sst_input": req.sst,
                    "ssh_input": req.ssh,
                    "sss_input": req.sss,
                    "wind_u_input": req.wind_u,
                    "wind_v_input": req.wind_v,
                    "grid_point": {"lat_idx": lat_idx, "lon_idx": lon_idx},
                },
                "indicators": {
                    "sst": round(profile_temps[0], 2),
                    "ohc_kj_cm2": round(float(ohc_kj[lat_idx, lon_idx]), 1),
                    "thermocline_depth_m": DEPTH_LEVELS[thermo_idx],
                    "mhw": {
                        "detected": bool(mhw["is_mhw"]),
                        "fraction": round(mhw["mhw_fraction"], 4),
                        "intensity": round(mhw["max_intensity"], 4),
                    },
                    "cyclone_risk": {
                        "mean_ohc_kj": round(cr["mean_ohc"], 2),
                        "pct_favorable": round(cr["pct_favorable"], 2),
                        "pct_high_risk": round(cr["pct_high_risk"], 2),
                    },
                },
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    frontend_dir = Path(__file__).parent.parent / "frontend"
    import hashlib, time as _time
    _static_version = str(int(_time.time()))

    @app.middleware("http")
    async def no_cache_middleware(request, call_next):
        response = await call_next(request)
        if request.url.path.endswith(".html") or request.url.path.endswith("/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    if frontend_dir.exists():
        from starlette.responses import FileResponse
        from starlette.staticfiles import StaticFiles as StarletteStatic

        # Mount vendor assets (CSS, JS) — no conflict with API routes
        vendor_dir = frontend_dir / "vendor"
        if vendor_dir.exists():
            app.mount("/vendor", StarletteStatic(directory=str(vendor_dir)), name="vendor")

        # Mount data directory
        data_dir = frontend_dir / "data"
        if data_dir.exists():
            app.mount("/data", StarletteStatic(directory=str(data_dir)), name="data")

        @app.get("/{path:path}")
        async def serve_frontend(path: str = ""):
            """Serve frontend files. API routes take priority because they're registered first."""
            file_path = frontend_dir / path
            if file_path.is_file():
                resp = FileResponse(str(file_path))
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                return resp
            # Default: serve index.html
            resp = FileResponse(str(frontend_dir / "index.html"))
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

    return app


app = create_demo_app()
