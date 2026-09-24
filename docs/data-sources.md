# Dataset Requirements & Download Plan

## Project: OceanEmbed — Subsurface Ocean Temperature Reconstruction
**SIH Problem Statement ID:** SIH26066 | **Team:** Neural Shadows
**Storage Budget:** 470 GB | **Domain:** North Indian Ocean (5°N–30°N, 45°E–105°E)

---

## 1. Dataset Summary Table

| # | Dataset | Variable(s) | Source | Resolution | Temporal | Time Range | Est. Size (Regional Subset) | Priority |
|---|---------|-------------|--------|------------|----------|------------|----------------------------|----------|
| 1 | **GLORYS12V1 Reanalysis** (Target/Label) | Temperature (thetao), Salinity (so), Currents (uo, vo), SSH (zos) | CMEMS `GLOBAL_MULTIYEAR_PHY_001_030` | 1/12° (~8km), 50 levels | Daily | 2010–2025 (16 yrs) | ~80–120 GB | **CRITICAL** |
| 2 | **OSTIA SST** (Input) | Sea Surface Temperature | CMEMS `SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001` | 0.05° (native), regrid to 0.25° | Daily | 2010–2025 | ~35 GB | **CRITICAL** |
| 3 | **DUACS SLA/SSH** (Input) | Sea Level Anomaly, Geostrophic Currents | CMEMS `SEALEVEL_GLO_PHY_L4_MY_008_047` | 0.25° | Daily | 2010–2025 | ~15 GB | **CRITICAL** |
| 4 | **SMOS/SMAP SSS** (Input) | Sea Surface Salinity | CMEMS `MULTIOBS_GLO_PHY_SSS_L4_MY_015_015` | 0.25° | Weekly | 2010–2025 | ~3–5 GB | **HIGH** |
| 5 | **Scatterometer Winds** (Input) | Surface Wind U, V | CMEMS `WIND_GLO_PHY_L3_MY_012_005` | 0.25° | Daily | 2010–2025 | ~18 GB | **HIGH** |
| 6 | **INCOIS Gridded Argo** (Validation) | Temperature, Salinity profiles | INCOIS LAS (`las.incois.gov.in`) | 1°×1°, 24 depth levels | Monthly & 10-day | 2010–2025 | ~2–3 GB | **CRITICAL** |
| 7 | **CMEMS SST L4 GMPE** (Alternative SST) | SST ensemble median | CMEMS `SST_GLO_PHY_L4_NRT_010_005` | 0.25° | Daily | 2010–2025 | ~10 GB | MEDIUM |

**Total Estimated Storage: ~165–210 GB** (leaves ~260–300 GB for processed data, models, and demos)

---

## 2. Detailed Dataset Descriptions

### 2.1 GLORYS12V1 Reanalysis (Training Target)
- **Full Name:** Global Ocean Physics Reanalysis
- **Product ID:** `GLOBAL_MULTIYEAR_PHY_001_030`
- **Provider:** Copernicus Marine Service (CMEMS) / Mercator Ocean
- **URL:** https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
- **Native Resolution:** 1/12° (~8km), 50 vertical levels
- **Variables Needed:**
  - `thetao` — Sea water potential temperature (°C) ← **primary training label**
  - `so` — Sea water salinity (PSU)
  - `uo` — Eastward sea water velocity (m/s)
  - `vo` — Northward sea water velocity (m/s)
  - `zos` — Sea surface height above geoid (m)
- **Time Range:** 1993-01-01 to present (updated yearly)
- **Download Strategy:** Subset spatially to North Indian Ocean (5°N–30°N, 45°E–105°E) + subset only upper 1000m (~15 of 50 levels) using CMEMS Subsetter or `motuclient`
- **Why Only 2010–2025:** Satellite inputs (SSS from SMAP starts 2015, scatterometer winds from Metop start 2007) — training period must overlap with available satellite inputs. 16 years × 365 days × ~daily files = ~5,840 files. Each regionally subsetted daily file with 5 variables at 15 depth levels ≈ 15–20 MB → **~90–120 GB total**
- **Access:** Free after CMEMS registration (https://marine.copernicus.eu)
- **Download Tool:** `motuclient` Python package or CMEMS web subsetter
- **Reference script:** https://github.com/gjpelletier/get_glorys

### 2.2 OSTIA SST (Input Variable #1)
- **Full Name:** Global Ocean OSTIA Sea Surface Temperature Analysis
- **Product ID:** `SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001`
- **Provider:** UK Met Office / CMEMS
- **URL:** https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/description
- **Native Resolution:** 0.05° × 0.05° (will regrid to 0.25°)
- **Variable:** `analysed_sst` (Foundation SST, Kelvin)
- **File Size:** ~15 MB per daily global file
- **Time Range:** 2006-12-31 to present
- **Download Strategy:** Subset to North Indian Ocean, daily, 2010–2025
- **Est. Regional Size:** ~35 GB (6 years × 365 days × ~10 MB after subsetting)
- **Access:** Free after CMEMS registration

### 2.3 DUACS SLA/SSH (Input Variable #2)
- **Full Name:** Global Ocean Gridded L4 Sea Surface Heights
- **Product ID:** `SEALEVEL_GLO_PHY_L4_MY_008_047` (multi-year) or `SEALEVEL_GLO_PHY_L4_NRT_008_046` (NRT)
- **Provider:** CLS / CMEMS
- **URL:** https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description
- **Resolution:** 0.25° × 0.25° (native!)
- **Variables:**
  - `sla` — Sea level anomaly (m)
  - `ugosa` — Geostrophic velocity anomaly: zonal (m/s)
  - `vgosa` — Geostrophic velocity anomaly: meridional (m/s)
- **Time Range:** 1993-01-01 to present
- **File Size:** ~4 MB per daily global file
- **Download Strategy:** Subset to North Indian Ocean, daily, 2010–2025
- **Est. Regional Size:** ~15 GB
- **Access:** Free after CMEMS registration

### 2.4 SMOS/SMAP SSS (Input Variable #3)
- **Full Name:** SSS SMOS/SMAP L4 OI — LOPS-v2025
- **Product ID:** `MULTIOBS_GLO_PHY_SSS_L4_MY_015_015`
- **Provider:** CATDS / CMEMS
- **URL:** https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_SSS_L4_MY_015_015/description
- **Resolution:** 0.25° × 0.25°
- **Variable:** `sss` — Corrected Practical Sea Surface Salinity (pss-78)
- **Temporal:** Weekly (7-day composites)
- **Time Range:** 2010-05-31 to present
- **Est. Regional Size:** ~3–5 GB (weekly files × 15 years)
- **Access:** Free after CMEMS registration

### 2.5 Scatterometer Winds (Input Variable #4)
- **Full Name:** Global Ocean Gridded L3 Sea Surface Winds from Scatterometer
- **Product ID:** `WIND_GLO_PHY_L3_MY_012_005`
- **Provider:** KNMI / CMEMS
- **URL:** https://data.marine.copernicus.eu/product/WIND_GLO_PHY_L3_MY_012_005/description
- **Resolution:** 0.25° × 0.25°
- **Variables:**
  - `eastward_wind` — Zonal wind (m/s)
  - `northward_wind` — Meridional wind (m/s)
- **Temporal:** Daily (ascending + descending passes)
- **Time Range:** 1994 to present (multi-mission)
- **Note:** Use Metop-A/B ASCAT for longest consistent record (2007+), or combine available scatterometers
- **Est. Regional Size:** ~18 GB (daily files × 16 years)
- **Access:** Free after CMEMS registration

### 2.6 INCOIS Gridded Argo (Validation)
- **Full Name:** Argo Value Added Products — Gridded Temperature & Salinity
- **Provider:** INCOIS (Indian National Centre for Ocean Information Services)
- **URL:** http://las.incois.gov.in/ (Live Access Server)
- **Alternative:** http://erddap.incois.gov.in/erddap/index.html
- **Resolution:** 1° × 1° (spatial), 24 standard depth levels
- **Temporal:** Monthly and 10-day composites
- **Time Range:** 2002 to present
- **Region:** Indian Ocean (30°E–120°E, 30°S–30°N)
- **Format:** NetCDF
- **Est. Size:** ~2–3 GB
- **Access:** Free, public access
- **Note:** This is our **independent validation** dataset — NOT used in training. We compare model-reconstructed profiles against Argo-measured profiles at held-out locations.

---

## 3. Storage Budget Breakdown

| Category | Est. Size |
|----------|-----------|
| GLORYS12 (raw regional subset) | 90–120 GB |
| OSTIA SST (raw regional subset) | 35 GB |
| DUACS SLA (raw regional subset) | 15 GB |
| SMOS/SMAP SSS (raw regional subset) | 3–5 GB |
| Scatterometer Winds (raw regional subset) | 18 GB |
| INCOIS Argo (validation) | 2–3 GB |
| **Raw data subtotal** | **~165–210 GB** |
| Processed/intermediate data | 50–80 GB |
| Model checkpoints (1–2 models) | 20–40 GB |
| Demo app + precomputed outputs | 30–50 GB |
| Notebooks, code, misc | 10–20 GB |
| **Grand Total** | **~275–400 GB** |
| **Remaining headroom** | **~70–195 GB** |

---

## 4. Download Method

All CMEMS datasets can be downloaded via:
1. **CMEMS Web Portal** (manual subsetter for small downloads)
2. **`motuclient`** Python package (automated bulk downloads)
3. **Copernicus Marine Toolbox** (new CLI/API)

### Setup Steps:
```bash
pip install motuclient
# Register at https://marine.copernicus.eu
# Store credentials in ~/.copernicusmarine/credentials or pass via CLI
```

### Example download command for GLORYS subset:
```bash
motuclient \
  --product-id GLOBAL_MULTIYEAR_PHY_001_030 \
  --dataset-id global-reanalysis-phy-001-030-daily \
  --date-min "2010-01-01" --date-max "2025-12-31" \
  --latitude-min 5 --latitude-max 30 \
  --longitude-min 45 --longitude-max 105 \
  --depth-min 0 --depth-max 1000 \
  --variable thetao --variable so --variable uo --variable vo --variable zos \
  --out-dir ../data/raw/glorys/ \
  --auth-mode=token
```

---

## 5. Important Notes

1. **All data is free** — CMEMS requires free registration only
2. **No raw data in git** — use `.gitignore` on `data/raw/` and `data/interim/`
3. **Document every download** — date, version, access method in `docs/references/`
4. **Regridding will happen in preprocessing** — OSTIA (0.05°→0.25°), GLORYS (1/12°→0.25°), winds (already 0.25°)
5. **Time range alignment** — All datasets need 2010–2025 overlap; SSS from SMOS starts 2010, SMAP starts 2015, so combined SSS product starts 2010
