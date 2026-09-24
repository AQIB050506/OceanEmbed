"""
Start OceanEmbed online demo server.
Usage: python demo/online/start.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    server = Path(__file__).parent / "server.py"
    print("=" * 60)
    print("OceanEmbed Real-Time Demo")
    print("=" * 60)
    print(f"Server: {server}")
    print(f"Frontend: http://localhost:8000")
    print()
    print("Features:")
    print("  - Fetches real-time SST from Open-Meteo Marine API")
    print("  - Fetches real-time wind from Open-Meteo Weather API")
    print("  - Runs model inference on GPU")
    print("  - Shows predicted subsurface temperature profiles")
    print()
    print("Starting server...")
    subprocess.run([sys.executable, str(server)], cwd=str(server.parent))

if __name__ == "__main__":
    main()
