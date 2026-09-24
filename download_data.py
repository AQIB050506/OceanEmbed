#!/usr/bin/env python3
"""
CMEMS Data Download Helper
Downloads satellite and reanalysis data from Copernicus Marine Service.
Requires: pip install motuclient
Register at: https://marine.copernicus.eu
"""
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

from src.config import (
    RAW_DIR, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    DEPTH_LEVELS,
)


def download_glorys(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    output_dir: Path = None,
):
    """Download GLORYS12V1 subset for North Indian Ocean."""
    output_dir = output_dir or RAW_DIR / "glorys"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "motuclient",
        "--product-id", "GLOBAL_MULTIYEAR_PHY_001_030",
        "--dataset-id", "global-reanalysis-phy-001-030-daily",
        "--date-min", start_date,
        "--date-max", end_date,
        "--latitude-min", str(LAT_MIN),
        "--latitude-max", str(LAT_MAX),
        "--longitude-min", str(LON_MIN),
        "--longitude-max", str(LON_MAX),
        "--depth-min", "0",
        "--depth-max", "1000",
        "--variable", "thetao",
        "--variable", "so",
        "--variable", "uo",
        "--variable", "vo",
        "--variable", "zos",
        "--out-dir", str(output_dir),
        "--auth-mode", "token",
    ]

    print(f"Downloading GLORYS to {output_dir}...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def download_sst(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    output_dir: Path = None,
):
    """Download OSTIA SST subset."""
    output_dir = output_dir or RAW_DIR / "sst"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "motuclient",
        "--product-id", "SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001",
        "--dataset-id", "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        "--date-min", start_date,
        "--date-max", end_date,
        "--latitude-min", str(LAT_MIN),
        "--latitude-max", str(LAT_MAX),
        "--longitude-min", str(LON_MIN),
        "--longitude-max", str(LON_MAX),
        "--variable", "analysed_sst",
        "--out-dir", str(output_dir),
        "--auth-mode", "token",
    ]

    print(f"Downloading SST to {output_dir}...")
    subprocess.run(cmd, check=True)


def download_sla(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    output_dir: Path = None,
):
    """Download DUACS SLA/SSH."""
    output_dir = output_dir or RAW_DIR / "sla"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "motuclient",
        "--product-id", "SEALEVEL_GLO_PHY_L4_MY_008_047",
        "--dataset-id", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
        "--date-min", start_date,
        "--date-max", end_date,
        "--latitude-min", str(LAT_MIN),
        "--latitude-max", str(LAT_MAX),
        "--longitude-min", str(LON_MIN),
        "--longitude-max", str(LON_MAX),
        "--variable", "sla",
        "--variable", "ugosa",
        "--variable", "vgosa",
        "--out-dir", str(output_dir),
        "--auth-mode", "token",
    ]

    print(f"Downloading SLA to {output_dir}...")
    subprocess.run(cmd, check=True)


def download_sss(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    output_dir: Path = None,
):
    """Download SMOS/SMAP SSS."""
    output_dir = output_dir or RAW_DIR / "sss"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "motuclient",
        "--product-id", "MULTIOBS_GLO_PHY_SSS_L4_MY_015_015",
        "--dataset-id", "cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c",
        "--date-min", start_date,
        "--date-max", end_date,
        "--latitude-min", str(LAT_MIN),
        "--latitude-max", str(LAT_MAX),
        "--longitude-min", str(LON_MIN),
        "--longitude-max", str(LON_MAX),
        "--variable", "sss",
        "--out-dir", str(output_dir),
        "--auth-mode", "token",
    ]

    print(f"Downloading SSS to {output_dir}...")
    subprocess.run(cmd, check=True)


def download_winds(
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    output_dir: Path = None,
):
    """Download scatterometer winds."""
    output_dir = output_dir or RAW_DIR / "winds"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "motuclient",
        "--product-id", "WIND_GLO_PHY_L3_MY_012_005",
        "--dataset-id", "cmems_obs-wind_glo_phy_my_l3-metopa-ascat-asc-0.25deg_P1D-i",
        "--date-min", start_date,
        "--date-max", end_date,
        "--latitude-min", str(LAT_MIN),
        "--latitude-max", str(LAT_MAX),
        "--longitude-min", str(LON_MIN),
        "--longitude-max", str(LON_MAX),
        "--variable", "eastward_wind",
        "--variable", "northward_wind",
        "--out-dir", str(output_dir),
        "--auth-mode", "token",
    ]

    print(f"Downloading winds to {output_dir}...")
    subprocess.run(cmd, check=True)


def download_all(start_date: str = "2010-01-01", end_date: str = "2025-12-31"):
    """Download all datasets."""
    print("=" * 60)
    print("OceanEmbed - CMEMS Data Download")
    print("=" * 60)

    print("\n[1/5] Downloading GLORYS12V1 reanalysis...")
    download_glorys(start_date, end_date)

    print("\n[2/5] Downloading OSTIA SST...")
    download_sst(start_date, end_date)

    print("\n[3/5] Downloading DUACS SLA...")
    download_sla(start_date, end_date)

    print("\n[4/5] Downloading SMOS/SMAP SSS...")
    download_sss(start_date, end_date)

    print("\n[5/5] Downloading Scatterometer Winds...")
    download_winds(start_date, end_date)

    print("\n" + "=" * 60)
    print("All downloads complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--dataset", choices=["all", "glorys", "sst", "sla", "sss", "winds"], default="all")
    args = parser.parse_args()

    if args.dataset == "all":
        download_all(args.start_date, args.end_date)
    elif args.dataset == "glorys":
        download_glorys(args.start_date, args.end_date)
    elif args.dataset == "sst":
        download_sst(args.start_date, args.end_date)
    elif args.dataset == "sla":
        download_sla(args.start_date, args.end_date)
    elif args.dataset == "sss":
        download_sss(args.start_date, args.end_date)
    elif args.dataset == "winds":
        download_winds(args.start_date, args.end_date)
