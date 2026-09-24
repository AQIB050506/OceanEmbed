"""
Offline demo server for OceanEmbed.
Serves the precomputed demo data and frontend.
Usage: python demo/serve_offline.py
"""
import http.server
import socketserver
import json
from pathlib import Path

PORT = 8000
DIRECTORY = Path(__file__).parent / "frontend"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[Server] {args[0]}")

def main():
    print(f"OceanEmbed Offline Demo Server")
    print(f"Serving from: {DIRECTORY}")
    print(f"Open: http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop\n")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
