import subprocess
import sys
import os

YEARS = list(range(2010, 2026))

DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs_P1D"
VARIABLES = ["sla", "ugosa", "vgosa"]
OUTPUT_DIR = "./data/sla"

LON_MIN, LON_MAX = 45, 100
LAT_MIN, LAT_MAX = 0, 25

os.makedirs(OUTPUT_DIR, exist_ok=True)

for year in YEARS:
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    filename = f"SLA_NIO_{year}.nc"

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
    print(f"Downloading SLA {year} -> {filename}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nFAILED for {year}. Stopping. Re-run from this year.")
        sys.exit(1)
    else:
        print(f"\n{year} complete.")

print("\nAll SLA years downloaded!")
