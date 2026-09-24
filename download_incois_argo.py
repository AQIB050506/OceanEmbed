
"""
Download INCOIS Gridded Argo Data via ERDDAP API
Covers 2010-2025 for North Indian Ocean (0-25°N, 45-100°E)
"""
 
import requests
import os
from datetime import datetime
import urllib3
 
# Suppress SSL certificate warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
# ERDDAP endpoint and dataset
ERDDAP_BASE = "https://erddap.incois.gov.in/erddap"
DATASET_ID = "incois_argo_mnt_VAM"  # Monthly Variational Analysis Methodology (includes T, S at depth)
 
# Region of interest (North Indian Ocean)
LON_MIN, LON_MAX = 45, 100
LAT_MIN, LAT_MAX = 0, 25
 
# Time range: 2010-01-15 to 2025-12-15 (monthly data, so mid-month dates)
TIME_START = "2010-01-15T00:00:00Z"
TIME_END = "2025-12-15T00:00:00Z"
 
# Output directory
OUTPUT_DIR = "./data/incois_argo"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# Variables available in INCOIS Argo: temperature, salinity (at multiple depths)
VARIABLES = ["temperature", "salinity"]
 
print("=" * 70)
print(f"Downloading INCOIS Gridded Argo Data")
print(f"Dataset: {DATASET_ID}")
print(f"Region: {LAT_MIN}°N-{LAT_MAX}°N, {LON_MIN}°E-{LON_MAX}°E (North Indian Ocean)")
print(f"Time: {TIME_START} to {TIME_END}")
print("=" * 70)
 
# Construct ERDDAP griddap URL for NetCDF download
# Format: https://erddap.incois.gov.in/erddap/griddap/{dataset}.nc?var1[time][lat][lon],var2[time][lat][lon],...
 
url = (
    f"{ERDDAP_BASE}/griddap/{DATASET_ID}.nc?"
    f"temperature[({TIME_START}):1:({TIME_END})][({LAT_MIN}):1:({LAT_MAX})][({LON_MIN}):1:({LON_MAX})],"
    f"salinity[({TIME_START}):1:({TIME_END})][({LAT_MIN}):1:({LAT_MAX})][({LON_MIN}):1:({LON_MAX})]"
)
 
print(f"\nConstructed ERDDAP URL:")
print(f"{url[:120]}...\n")
 
output_file = os.path.join(OUTPUT_DIR, "INCOIS_ARGO_NorthIndianOcean_2010-2025.nc")
 
try:
    print(f"Downloading to: {output_file}")
    print("(This may take a few minutes...)")
    
    response = requests.get(url, stream=True, timeout=600, verify=False)
    response.raise_for_status()
    
    total_size = response.headers.get('content-length')
    if total_size:
        total_size = int(total_size) / (1024 ** 3)  # Convert to GB
        print(f"Estimated size: ~{total_size:.2f} GB")
    
    # Write to file with progress
    with open(output_file, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / (int(response.headers['content-length']))) * 100
                    print(f"\rProgress: {pct:.1f}%", end='', flush=True)
    
    print(f"\n\n✅ Download successful!")
    print(f"File: {output_file}")
    print(f"Size: {os.path.getsize(output_file) / (1024**3):.2f} GB")
 
except requests.exceptions.RequestException as e:
    print(f"\n❌ Download failed: {e}")
    print(f"Troubleshooting:")
    print(f"1. Check ERDDAP service at: https://erddap.incois.gov.in/erddap/index.html")
    print(f"2. Verify dataset '{DATASET_ID}' exists")
    print(f"3. Try manually downloading from: https://las.incois.gov.in/las/")
 
print("\n" + "=" * 70)
 