import subprocess
import sys
import os

YEARS = list(range(2010, 2026))

DATASET_ID = "cmems_obs-wind_glo_phy_my_l3-metopa-ascat-asc-0.25deg_P1D"
VARIABLES = ["eastward_wind", "northward_wind"]
OUTPUT_DIR = "./data/winds"

LON_MIN, LON_MAX = 45, 100
LAT_MIN, LAT_MAX = 0, 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

for year in YEARS:
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    filename = f"WINDS_NIO_{year}.nc"

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id", DATASET_ID,
        "--start-datetime", start,
        "--end-datetime",   end,
        "--minimum-longitude", str(LON_MIN),
        "--maximum-longitude", str(LON_MAX),
        "--minimum-latitude",  str(LAT_MIN),
        "--maximum-latitude",  str(LAT_MAX),
        "--output-filename",   filename,
        "--output-directory",  OUTPUT_DIR,
    ]
    for v in VARIABLES:
        cmd += ["--variable", v]

    print(f"\n{'='*60}")
    print(f"Downloading Winds {year} -> {filename}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nFAILED for {year}. Stopping. Re-run from this year.")
        sys.exit(1)
    else:
        print(f"\n{year} complete.")

print("\nAll Winds years downloaded!")
