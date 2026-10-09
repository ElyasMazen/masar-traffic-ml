# Masar Pilot ML Package

This package is the pilot ML/training delivery for **Masar — Smart Traffic Intelligence**.

## Fastest way to use it

### 1) Create environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Re-train on a downloaded snapshot

```powershell
python train_masar.py --csv "jeddah_tomtom_flow 10.10(2).csv" --out retrain_outputs
```

**Never point the scripts at the live GitHub workflow and never call the TomTom API.** Work only on downloaded CSV snapshots.

### 3) Predict the next ~10-minute congestion state

Using the packaged full-pilot demo model:

```powershell
python predict_next.py --csv "jeddah_tomtom_flow 10.10(2).csv" --model "models/random_forest_pilot_fullfit.joblib" --out next_step_forecast.csv
```

## Scientific rule used here

The target label is derived from the current row's `speed_ratio`, so current-row traffic values are **not** allowed as predictors. The model uses past observations only.

## Headline pilot result

Selected ML model: **Random Forest**

Chronological test:
- Random Forest Accuracy: **0.9570**
- Random Forest Macro-F1: **0.8873**
- Persistence Macro-F1: **0.8890**

These are **pilot-only** results. See `PILOT_REPORT.md` for limitations and suspicious-data notes.
