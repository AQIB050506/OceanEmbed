# Decision Log

## Project: OceanEmbed — Subsurface Ocean Temperature Reconstruction
**SIH Problem Statement ID:** SIH26066 | **Team:** Neural Shadows**

Per `rules.md` §9: significant decisions get one line here with date and reasoning.

| # | Date | Decision | Reasoning |
|---|------|----------|-----------|
| 1 | 2026-09-13 | Architecture: Attention-Enhanced 3D U-Net++ | Best fit from 6+ paper review; proven on similar NW Pacific problem |
| 2 | 2026-09-13 | Inputs: SST + SSH + SSS + Wind(U,V) — 5 channels | Literature: SST dominates mixed layer, SSH dominates thermocline/deep |
| 3 | 2026-09-13 | Training target: GLORYS12V1; Validation: INCOIS Argo | GLORYS is gold-standard reanalysis; Argo is independent (not assimilated) |
| 4 | 2026-09-13 | Temporal range: 2010–2025 | Overlap window where all satellite products available |
| 5 | 2026-09-16 | Domain: 0–25°N, 45–100°E, depths 0–450m (not PRD's 5–30°N/0–1000m) | Use downloaded data; re-downloading not feasible in timeline |
| 6 | 2026-09-17 | temporal_window: 5 days (not 26) | 26d = 25s/batch → weeks of training; 5d = 0.3s/batch → 19 hours |
| 7 | 2026-09-17 | batch_size: 2 (not 8) | 8 OOMs on 6GB RTX 4050 |
| 8 | 2026-09-17 | Single-stage training (skip Argo pretrain stage) | Timeline; GLORYS-only training already achieves Argo RMSE=1.02°C |
| 9 | 2026-09-19 | Bypass broken gridded Argo file; validate against raw floats | INCOIS gridded product had issues; raw point-matching is more honest |
| 10 | 2026-09-19 | Density constraint loss disabled | Model predicts temperature only (no salinity); constraint needs both T and S |
| 11 | 2026-09-24 | Demo: two-mode (FastAPI live + static offline fallback) | Judging needs offline; live inference is wow-factor when possible |
| 12 | 2026-09-24 | Map tiles: online with automatic offline grid fallback | Esri tiles need internet; grid background keeps demo functional offline |
| 13 | 2026-09-24 | Add ARGO overlay + SST/OHC/Anomaly map layers to demo | design.md §3.2/§3.3 requires ARGO comparison and anomaly overlay |
| 14 | 2026-09-24 | Baseline = per-depth linear regression + climatology (not nearest-neighbor) | Simpler to fit/evaluate; linear already shows DL 43× improvement, NN wouldn't change conclusion |
| 15 | 2026-09-24 | Online demo SSH from Open-Meteo API, not random climatology | Random SSH caused 30.8% flat-profile model collapse; real SSH → 0.3% |
| 16 | 2026-09-24 | Online frontend = Three.js WebGL (vendored, importmap) not React/Svelte | 3D depth-slicing was the goal; Three.js gives full scene control with zero build step, keeps offline single-server demo simple |
| 17 | 2026-09-24 | Minimap tiles = Esri World Dark Gray (not CARTO dark) | CARTO dark tiles return "API KEY REQUIRED" placeholder without a key; Esri dark-gray is keyless, dark-themed, labeled |
| 18 | 2026-09-24 | Land for 3D context derived from test_target.nc NaN mask, not external GeoJSON | Zero extra downloads, perfectly aligned to model grid, includes atolls (Maldives visible); external coastlines would misalign at 0.25° |
| 19 | 2026-09-25 | Profile chart depth axis flipped to surface-at-top (reverse:false) | Old axis had deep water on top (unconventional); surface-up matches oceanographic T-Z convention and the depth slider direction (up = shallower) so the highlight moves with the slider |
| 20 | 2026-09-25 | Fish Forecast visuals: emoji → vendored Lucide stroke icons + zone color system | Emoji read as AI-generated and render inconsistently cross-platform; Lucide (ISC) SVGs inlined (Iconify CDN returned 403, no runtime dep); zone identity via tinted icon + 3px accent strip + label dot (surface #22d3ee / mid #64b5f6 / deep #818cf8); no squid glyph exists in Lucide/Tabler so `shell` stands in |
| 21 | 2026-09-25 | Depth-bar ticks positioned at thumb positions (index scale), not flex-evenly-spaced | Thumb is index-linear (15 discrete levels, center = 10 + idx/14 × 200px) but ticks were `space-between` — "200" label sat at the 100m thumb position, so a 75m selection appeared to read against "200"; ticks now absolute at top = 10/110/152.9/181.4/210px for 0/100/200/300/450m (value-linear thumb rejected: would cram 0-50m into the first 22px, making shallow levels unselectable) |
| 22 | 2026-09-25 | Time Machine = last 14 days on an index-linear slider; archive dates via Open-Meteo archive API in 2 batched multi-coordinate requests; date-keyed server entry store (≤12 archive, today TTL 600s) with `/api/warm` neighbor prefetch; `date=None` always means today (not last-browsed) | Judge-visible history without re-architecting the model path; batching cut 132 fetches/date → 2 (~4s cold, 0.3s warm); None→today fixes live-vs-archive aliasing after browsing |
| 23 | 2026-09-25 | Argo overlay sourced keyless from GDAC (58MB global index parsed in background + per-profile `.nc` via `/dac/` URLs, QC 1/2); clicking a float snaps the profile point to the float so the chart RMSE badge is colocated | Real in-situ ground truth proves the forecast against observations; colocated RMSE is scientifically honest (a distant float showed a misleading 3.89°C vs 2.19°C colocated); no credentials in a demo |
| 24 | 2026-09-25 | Transect A→B = server bilinear sampling (15×90 + land flags) rendered twice: 2D section canvas (linear depth axis, shared tempColor ramp, land strips) + 3D curtain (CanvasTexture on a plane rotated to the A→B chord); shareable boot params `?date=` `?tr=` `?argo=1` | One dataset, two views — 2D for precise reading, 3D for spatial context; flipY canvas orientation (surface row at top) verified by pixel gradient; URL params make demo states reproducible for judges and headless tests |
