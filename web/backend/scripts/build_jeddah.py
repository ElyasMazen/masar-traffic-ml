"""
Prepare the Jeddah view: (re)train with the repo's train_masar.py and check
that everything the backend loads is present.

    python scripts\\build_jeddah.py                   # check; trains only if something is missing
    python scripts\\build_jeddah.py --retrain         # retrain on the raw GitHub CSV
    python scripts\\build_jeddah.py --retrain --csv snapshot.csv

The web app lives in masar-traffic-ml/web and reads models/, outputs/results.json
and the training snapshot (data/snapshots/, gitignored) directly from the repo
root - nothing is copied. Never calls the TomTom API and never writes to the data repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]   # web/backend/scripts -> repo root
RESULTS = REPO_ROOT / "outputs" / "results.json"


def repo_python() -> str:
    """Prefer the repo's own environment (it has matplotlib for the charts)."""
    venv = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv) if venv.exists() else sys.executable


def missing_artifacts() -> list[str]:
    if not RESULTS.exists():
        return ["outputs/results.json"]
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    horizon = results["short_term"]["selected"]["horizon_min"]
    needed = [
        f"models/short_term_rf_{horizon}min.joblib",
        "models/time_of_day.joblib",
        results["data"]["snapshot"] or "data/snapshots/<training snapshot>",
    ]
    return [p for p in needed if not (REPO_ROOT / p).exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="Run train_masar.py even if artifacts exist")
    ap.add_argument("--csv", default=None, help="With --retrain: local snapshot instead of the raw GitHub CSV")
    args = ap.parse_args()

    missing = missing_artifacts()
    if args.retrain or missing:
        if missing and not args.retrain:
            print(f"Missing {', '.join(missing)} - training first.")
        cmd = [repo_python(), "train_masar.py"] + (["--csv", str(Path(args.csv).resolve())] if args.csv else [])
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        missing = missing_artifacts()
    if missing:
        raise SystemExit(f"Still missing after training: {', '.join(missing)}")

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    selected = results["short_term"]["selected"]
    print(f"Models trained through {results['data']['trained_through']} ({results['data']['days']} days)")
    print(f"Short-term horizon shown in the app: +{selected['horizon_min']} min")
    print(f"  {selected['sentence']}")
    print(f"Time-of-day method: {results['time_of_day']['selected']['label']}")
    print("Restart the backend to load them.")


if __name__ == "__main__":
    main()
