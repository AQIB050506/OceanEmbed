import subprocess
import sys

YEARS = [2026]  # 2010 already done

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = ["thetao", "so"]
OUTPUT_DIR = "./data/glorys"

LON_MIN, LON_MAX = 45, 100
LAT_MIN, LAT_MAX = 0, 25
DEPTH_MIN, DEPTH_MAX = 0, 500

for year in YEARS:
    start = f"{year}-01-01"
    end   = f"{year}-08-31"
    filename = f"GLORYS_NIO_thetao_so_{year}.nc"

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id", DATASET_ID,
        "--start-datetime", start,
        "--end-datetime",   end,
        "--minimum-longitude", str(LON_MIN),
        "--maximum-longitude", str(LON_MAX),
        "--minimum-latitude",  str(LAT_MIN),
        "--maximum-latitude",  str(LAT_MAX),
        "--minimum-depth",     str(DEPTH_MIN),
        "--maximum-depth",     str(DEPTH_MAX),
        "--output-filename",   filename,
        "--output-directory",  OUTPUT_DIR,
    ]
    for v in VARIABLES:
        cmd += ["--variable", v]

    print(f"\n{'='*60}")
    print(f"Downloading {year} → {filename}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ FAILED for {year}. Stopping. Re-run from this year.")
        sys.exit(1)
    else:
        print(f"\n✅ {year} complete.")

print("\n🎉 All years downloaded successfully!")
