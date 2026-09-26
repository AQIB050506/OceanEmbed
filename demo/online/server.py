"""
Online demo backend for OceanEmbed.
Fetches real-time satellite data, runs model inference, serves results.
Adds time-playback (historical dates), Argo overlay, and transect endpoints.
"""
import numpy as np
import json
import time
import re
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import (
    MODELS_DIR, PROCESSED_DIR, DEPTH_LEVELS,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION,
)
from src.application.products import (
    compute_ocean_heat_content,
    detect_marine_heatwave,
    classify_cyclone_risk,
)
from realtime_fetcher import fetch_realtime_data, fetch_data_for_date

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).parent / "assets"
ONNX_MODEL_PATH = ASSETS_DIR / "model.onnx"

PLAYBACK_DAYS = 14
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ARGO_FILE_RE = re.compile(r"^[A-Za-z0-9_\-]+/\d+/profiles/[A-Za-z]\d+_\d+\.nc$")


class RealtimeInferenceEngine:
    """ONNX Runtime when assets/model.onnx exists (Vercel + local default),
    torch fallback otherwise (original GPU path)."""

    def __init__(self):
        self.model = None
        self.device = None
        self.session = None
        self.norm_stats = None

    @staticmethod
    def _load_norm_stats():
        for p in (ASSETS_DIR / "norm_stats.json", PROCESSED_DIR / "normalization_stats.json"):
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        raise FileNotFoundError("normalization_stats.json not found")

    def load_model(self):
        self.norm_stats = self._load_norm_stats()
        if ONNX_MODEL_PATH.exists():
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"]
            )
            logger.info("ONNX model loaded (%.1f MB)", ONNX_MODEL_PATH.stat().st_size / 1e6)
            return
        import torch
        from src.reconstruction.model import AttentionUNet3D
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(MODELS_DIR / "best_model.pt", map_location=self.device, weights_only=False)
        self.model = AttentionUNet3D()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)
        logger.info("Model loaded (epoch %d)", ckpt.get("epoch", -1))

    def predict(self, model_input):
        x = np.asarray(model_input, dtype=np.float32)
        if self.session is not None:
            pred = self.session.run(None, {"input": x[None]})[0][0]
        else:
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
                pred = self.model(tensor).cpu().numpy()[0]
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


def load_land_mask():
    """Land mask (True=land): precomputed npy (deploy) or test_target.nc NaNs (local)."""
    npy = ASSETS_DIR / "land_mask.npy"
    if npy.exists():
        return np.load(npy)
    import xarray as xr
    with xr.open_dataset(PROCESSED_DIR / "test_target.nc") as ds:
        t = ds["thetao"].isel(time=0).values
    mask = np.isnan(t).any(axis=0)  # (101, 221)
    mask = np.nan_to_num(mask.astype(np.float32))
    return mask


def today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def validate_date(date_str):
    """Accept YYYY-MM-DD within the playback window (slight future margin not allowed)."""
    if not date_str or not DATE_RE.match(date_str):
        return False
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return False
    t = datetime.now(timezone.utc).date()
    return t - timedelta(days=PLAYBACK_DAYS + 6) <= d <= t


# ---------- Argo GDAC ----------

import os

if os.environ.get("VERCEL") or os.environ.get("ARGO_CACHE_DIR"):
    ARGO_CACHE_DIR = Path(os.environ.get("ARGO_CACHE_DIR", "/tmp/oceanembed_argo"))
else:
    ARGO_CACHE_DIR = Path(__file__).parent / "cache"
ARGO_INDEX_GZ = ARGO_CACHE_DIR / "argo_index.txt.gz"
ARGO_POINTS_CSV = ARGO_CACHE_DIR / "argo_points.csv"
ARGO_POINTS_FALLBACK = ASSETS_DIR / "argo_points.csv"
ARGO_PROF_DIR = ARGO_CACHE_DIR / "profiles"
ARGO_BASE = "https://data-argo.ifremer.fr/dac/"


def _read_points_csv(path):
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    points = []
    for r in rows:
        parts = str(r["file"]).split("/")
        if len(parts) < 3:
            continue
        wmo = parts[1]
        cyc = Path(parts[-1]).stem.split("_")[-1].lstrip("0") or "0"
        points.append({
            "f": str(r["file"]), "wmo": wmo, "cycle": cyc,
            "date": str(r["dt"]), "lat": float(r["latitude"]), "lon": float(r["longitude"]),
        })
    return points


def load_argo_points(max_points=500, max_age_days=180):
    """Recent in-domain profiles from the GDAC global index (cached subset CSV)."""
    ARGO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ARGO_POINTS_CSV.exists() and time.time() - ARGO_POINTS_CSV.stat().st_mtime < 7 * 86400:
        return _read_points_csv(ARGO_POINTS_CSV)
    if not ARGO_INDEX_GZ.exists():
        if ARGO_POINTS_FALLBACK.exists():
            return _read_points_csv(ARGO_POINTS_FALLBACK)
        raise FileNotFoundError("argo_index.txt.gz not downloaded")
    import pandas as pd
    df = pd.read_csv(ARGO_INDEX_GZ, comment="#", compression="gzip",
                     usecols=["file", "date", "latitude", "longitude", "ocean"],
                     dtype={"date": "str"})
    df = df[df["ocean"] == "I"]
    df = df[(df["latitude"] >= LAT_MIN) & (df["latitude"] <= LAT_MAX) &
            (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX)]
    dt = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")
    df = df.assign(_dt=dt).dropna(subset=["_dt"])
    cutoff = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=max_age_days))
    df = df[df["_dt"] >= cutoff].sort_values("_dt", ascending=False).head(max_points)
    df["dt"] = df["_dt"].dt.strftime("%Y-%m-%d")
    df[["file", "date", "latitude", "longitude", "dt"]].to_csv(ARGO_POINTS_CSV, index=False)
    return _read_points_csv(ARGO_POINTS_CSV)


def load_argo_profile(file_rel):
    """Download + parse one Argo profile NetCDF → QC'd depths/temps."""
    if not ARGO_FILE_RE.match(file_rel or ""):
        raise ValueError("bad profile path")
    import netCDF4 as nc4
    import requests
    ARGO_PROF_DIR.mkdir(parents=True, exist_ok=True)
    local = ARGO_PROF_DIR / file_rel.replace("/", "_")
    if not local.exists():
        r = requests.get(ARGO_BASE + file_rel, timeout=30)
        r.raise_for_status()
        local.write_bytes(r.content)
    with nc4.Dataset(local) as ds:
        v = ds.variables
        temp = np.ma.filled(np.ma.asarray(v["TEMP"][0, :]), np.nan).astype(float)
        qc = np.ma.asarray(v["TEMP_QC"][0, :]) if "TEMP_QC" in v else None
        if "DEPTH" in v:
            depth = np.ma.filled(np.ma.asarray(v["DEPTH"][0, :]), np.nan).astype(float)
        elif "PRES" in v:
            depth = np.ma.filled(np.ma.asarray(v["PRES"][0, :]), np.nan).astype(float)
        else:
            raise ValueError("no DEPTH/PRES in profile")
        cycle = None
        if "CYCLE_NUMBER" in v:
            cycle = int(np.ravel(np.ma.filled(v["CYCLE_NUMBER"][:], -1))[0])
        prof_date = None
        if "JULD" in v:
            juld = float(np.ravel(np.ma.filled(v["JULD"][:], np.nan))[0])
            if np.isfinite(juld) and 0 < juld < 100000:
                prof_date = (datetime(1950, 1, 1) + timedelta(days=juld)).strftime("%Y-%m-%d")
    mask = np.isfinite(temp) & np.isfinite(depth) & (depth >= 0)
    if qc is not None:
        q = np.array([str(x).strip() for x in np.asarray(qc).ravel()])
        good = np.isin(q, ("1", "2"))
        if good.any():
            mask = mask & good
    if not mask.any():
        raise ValueError("no valid levels")
    d, t = depth[mask], temp[mask]
    order = np.argsort(d)
    d, t = d[order], t[order]
    if len(d) > 400:
        idx = np.linspace(0, len(d) - 1, 400).astype(int)
        d, t = d[idx], t[idx]
    return {
        "depths": [round(float(x), 1) for x in d],
        "temperature": [round(float(x), 2) for x in t],
        "n": int(len(d)),
        "cycle": cycle,
        "date": prof_date,
    }


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def sample_transect(preds, land, lat0, lon0, lat1, lon1, n):
    """Bilinear samples of (15,101,221) along the A→B line → dicts for section rendering."""
    n = max(20, min(int(n), 200))
    lats = np.linspace(lat0, lat1, n)
    lons = np.linspace(lon0, lon1, n)
    n_d = preds.shape[0]
    temps = np.full((n_d, n), np.nan)
    is_land = np.zeros(n, dtype=np.int8)
    dist = [0.0]
    for k in range(1, n):
        dist.append(dist[-1] + float(haversine_km(lats[k - 1], lons[k - 1], lats[k], lons[k])))
    for i in range(n):
        fi = float(np.clip((lats[i] - LAT_MIN) / RESOLUTION, 0, preds.shape[1] - 1.001))
        fj = float(np.clip((lons[i] - LON_MIN) / RESOLUTION, 0, preds.shape[2] - 1.001))
        i0, j0 = int(np.floor(fi)), int(np.floor(fj))
        di, dj = fi - i0, fj - j0
        if land is not None:
            li = min(int(round(fi)), land.shape[0] - 1)
            lj = min(int(round(fj)), land.shape[1] - 1)
            if land[li, lj] > 0.5:
                is_land[i] = 1
        for d in range(n_d):
            f = preds[d]
            v = (f[i0, j0] * (1 - di) * (1 - dj) + f[i0 + 1, j0] * di * (1 - dj)
                 + f[i0, j0 + 1] * (1 - di) * dj + f[i0 + 1, j0 + 1] * di * dj)
            temps[d, i] = v
    return {
        "distances_km": [round(x, 2) for x in dist],
        "temps": [[None if (is_land[i] or not np.isfinite(v)) else round(float(v), 2)
                   for i, v in enumerate(row)] for row in temps],
        "land": is_land.tolist(),
        "total_km": round(dist[-1], 1),
    }


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
    state = {
        "land": None,
        "entries": {},   # date -> {"data","preds","ts","inference_s"}
        "latest": None,  # date key used when no explicit date given
        "argo": {"points": None, "by_file": {}, "ready": False, "error": None},
    }

    @app.on_event("startup")
    async def startup():
        engine.load_model()
        try:
            state["land"] = load_land_mask()
            logger.info("Land mask loaded: %.1f%% land", 100 * state["land"].mean())
        except Exception as e:
            logger.warning("Land mask unavailable: %s", e)

        import threading

        def _argo_load():
            try:
                pts = load_argo_points()
                state["argo"]["points"] = pts
                state["argo"]["by_file"] = {p["f"]: p for p in pts}
                state["argo"]["ready"] = True
                logger.info("Argo index: %d recent in-domain profiles", len(pts))
            except Exception as e:
                state["argo"]["error"] = str(e)
                logger.warning("Argo index load failed: %s", e)

        threading.Thread(target=_argo_load, daemon=True).start()

    def get_entry(date_str):
        """Fetch + predict for a date on demand. Returns (entry, error)."""
        if date_str is None:
            date_str = today_str()
        if date_str is None:
            return None, "No data yet. Fetch current conditions first."
        if not validate_date(date_str):
            return None, f"Date out of range (allowed: last {PLAYBACK_DAYS} days)"
        now = time.time()
        is_today = date_str == today_str()
        ent = state["entries"].get(date_str)
        if ent and not (is_today and (now - ent["ts"]) > 600):
            return ent, None
        data = fetch_realtime_data() if is_today else fetch_data_for_date(date_str)
        t0 = time.time()
        preds = engine.predict(data["model_input"])
        inference_s = round(time.time() - t0, 2)
        ent = {"data": data, "preds": preds, "ts": time.time(), "inference_s": inference_s}
        state["entries"][date_str] = ent
        state["latest"] = date_str
        archive = sorted((k for k in state["entries"] if k != today_str()), reverse=True)
        for k in archive[11:]:
            del state["entries"][k]
        return ent, None

    def entry_or_error(date_str):
        ent, err = get_entry(date_str)
        if ent is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=err)
        return ent

    class ProfileQuery(BaseModel):
        latitude: float
        longitude: float
        date: str | None = None

    class TransectQuery(BaseModel):
        lat0: float
        lon0: float
        lat1: float
        lon1: float
        n: int = 80
        date: str | None = None

    @app.get("/api/realtime")
    def get_realtime():
        tdy = today_str()
        before = state["entries"].get(tdy)
        cached = before is not None and (time.time() - before["ts"]) < 600
        try:
            ent, err = get_entry(tdy)
            if err:
                return {"error": err}
            sst = ent["data"]["sst"]
            return {
                "date": tdy,
                "from_cache": cached,
                "inference_time_s": None if cached else ent["inference_s"],
                "sst_mean": round(float(np.nanmean(sst)), 2),
                "message": "Cached data" if cached else "Fresh prediction",
                "sources": ent["data"].get("sources"),
            }
        except Exception as e:
            logger.error("Realtime failed: %s", e)
            return {"error": str(e)}

    @app.get("/api/warm")
    def warm(date: str):
        """Build an archive entry without transferring the volume (playback prefetch)."""
        if not validate_date(date):
            return {"error": "bad date"}
        try:
            ent, err = get_entry(date)
            if err:
                return {"error": err}
            return {"date": date, "ready": True}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/profile")
    def get_profile(query: ProfileQuery):
        ent, err = get_entry(query.date)
        if ent is None:
            return {"error": err}
        preds = ent["preds"]
        lat_idx, lon_idx = latlon_to_grid(query.latitude, query.longitude)
        temps = preds[:, lat_idx, lon_idx].tolist()
        source = "model"

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

        mhw = detect_marine_heatwave(preds, DEPTH_LEVELS)

        ohc_full = compute_ocean_heat_content(
            np.transpose(preds, (1, 2, 0)), DEPTH_LEVELS, (0, 200)
        )
        cr = classify_cyclone_risk(ohc_full)

        return {
            "latitude": query.latitude,
            "longitude": query.longitude,
            "date": ent["data"]["date"],
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
    def get_sst_map():
        ent, err = get_entry(None)
        if ent is None:
            return {"error": err}
        sst = ent["data"]["sst"]
        return {
            "date": ent["data"]["date"],
            "sst": [[round(v, 2) for v in row] for row in sst.tolist()],
            "lats": [round(LAT_MIN + i * RESOLUTION, 2) for i in range(sst.shape[0])],
            "lons": [round(LON_MIN + j * RESOLUTION, 2) for j in range(sst.shape[1])],
        }

    @app.get("/api/pred_map")
    def get_pred_map(depth_idx: int = 0, date: str | None = None):
        ent, err = get_entry(date)
        if ent is None:
            return {"error": err}
        pred = ent["preds"][depth_idx]
        return {
            "depth": DEPTH_LEVELS[depth_idx],
            "temperature": [[round(v, 1) for v in row] for row in pred.tolist()],
        }

    @app.get("/api/volume")
    def get_volume(stride: int = 2, date: str | None = None):
        """
        All 15 depth layers downsampled by `stride` for the 3D viewer.
        Returns flattened Float32-style arrays (rounded to 1 decimal) per depth.
        """
        ent, err = get_entry(date)
        if ent is None:
            return {"error": err}
        preds = ent["preds"]  # (15, 101, 221)
        layers = []
        for d in range(preds.shape[0]):
            layer = preds[d, ::stride, ::stride]
            layers.append([round(float(v), 1) for v in layer.ravel()])
        h = preds.shape[1] // stride + (1 if preds.shape[1] % stride else 0)
        w = preds.shape[2] // stride + (1 if preds.shape[2] % stride else 0)
        sst = ent["data"]["sst"][::stride, ::stride]
        land = None
        if state["land"] is not None:
            land = [int(v) for v in state["land"].ravel()]
        return {
            "date": ent["data"]["date"],
            "depths": DEPTH_LEVELS,
            "n_lat": preds.shape[1],
            "n_lon": preds.shape[2],
            "stride": stride,
            "grid_h": h,
            "grid_w": w,
            "layers": layers,
            "sst": [round(float(v), 1) for v in sst.ravel()],
            "land": land,
            "land_h": state["land"].shape[0] if state["land"] is not None else None,
            "land_w": state["land"].shape[1] if state["land"] is not None else None,
            "bounds": {"lat_min": LAT_MIN, "lat_max": LAT_MAX, "lon_min": LON_MIN, "lon_max": LON_MAX},
        }

    @app.post("/api/transect")
    def post_transect(q: TransectQuery):
        """Temperature section along an arbitrary A→B line (15 depths × n points)."""
        for lat, lon in ((q.lat0, q.lon0), (q.lat1, q.lon1)):
            if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                return {"error": "endpoints outside domain"}
        ent, err = get_entry(q.date)
        if ent is None:
            return {"error": err}
        section = sample_transect(ent["preds"], state["land"], q.lat0, q.lon0, q.lat1, q.lon1, q.n)
        section.update({
            "date": ent["data"]["date"],
            "depths": DEPTH_LEVELS,
            "from": {"lat": q.lat0, "lon": q.lon0},
            "to": {"lat": q.lat1, "lon": q.lon1},
        })
        return section

    @app.get("/api/argo/points")
    def argo_points():
        a = state["argo"]
        if a["error"]:
            return {"error": a["error"]}
        if not a["ready"]:
            return {"ready": False, "points": []}
        return {"ready": True, "n": len(a["points"]), "points": a["points"]}

    @app.get("/api/argo/profile")
    def argo_profile(f: str):
        meta = state["argo"].get("by_file", {}).get(f)
        if meta is None:
            return {"error": "unknown profile (not in recent in-domain index)"}
        try:
            prof = load_argo_profile(f)
        except Exception as e:
            return {"error": str(e)}
        out = dict(meta)
        out.update({k: v for k, v in prof.items() if v is not None})
        if prof.get("cycle"):
            out["cycle"] = str(prof["cycle"])
        if prof.get("date"):
            out["date"] = prof["date"]
        return out

    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
