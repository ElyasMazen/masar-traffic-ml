from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
checks = [
    ("GET", "/health", None),
    ("GET", "/api/summary", None),
    ("GET", "/api/roads?limit=2", None),
    ("GET", "/api/roads/map", None),
    ("GET", "/api/segments", None),
    ("GET", "/api/government/priority?limit=2", None),
    ("POST", "/api/police/snapshot", {"date": "2026-08-19", "hour": 18}),
]

failed = False
for method, path, payload in checks:
    response = client.get(path) if method == "GET" else client.post(path, json=payload)
    mark = "OK" if response.status_code == 200 else "FAIL"
    print(f"[{mark}] {method} {path} -> {response.status_code}")
    failed = failed or response.status_code != 200

raise SystemExit(1 if failed else 0)
