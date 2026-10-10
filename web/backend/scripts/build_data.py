"""
Optional rebuild script.

The downloadable project already includes processed data and a trained model.
Use this script only if you want to regenerate the processed files from the
original NYC CSV files.

Place:
  nyc_traffic_volume.csv
  nyc_accidents_small.csv
inside backend/data/raw/

This script intentionally points you to the project README for the full
rebuild workflow, because the raw crash file is very large.
"""

from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

traffic = RAW / "nyc_traffic_volume.csv"
accidents = RAW / "nyc_accidents_small.csv"

print("Raw data directory:", RAW)
print("Traffic file:", "FOUND" if traffic.exists() else "MISSING")
print("Accident file:", "FOUND" if accidents.exists() else "MISSING")
print()
print("The shipped project is already ready to run using backend/data/processed.")
print("Keep the raw CSV files here if you want to extend/retrain the project later.")
