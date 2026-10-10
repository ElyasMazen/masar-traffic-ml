# Masar Pilot ML Package

This package is the pilot ML/training delivery for **Masar — Smart Traffic Intelligence**.

## Fastest way to use it

### 1) Create environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Train and evaluate

```powershell
python train_masar.py                                  # latest data from the raw GitHub CSV
python train_masar.py --csv "data\snapshots\<file>.csv" # re-run on a saved snapshot
```

By default the data is **read** from the raw GitHub CSV of the collector repo
(`https://raw.githubusercontent.com/elAmigoOs/tomtomdata/main/jeddah_tomtom_flow.csv`).
Each download is saved to `data/snapshots/` as a dated snapshot, so every result can be reproduced with `--csv`.
raw.githubusercontent.com can lag about 5 minutes behind the latest commit.

**Never call the TomTom API from this repo, and never write to the data repo.** Reading the raw CSV is the only access.

### 3) Forecast the next 10 / 30 / 60 minutes

```powershell
python predict_next.py --horizon 60
```

## What gets written

- `outputs/results.json` — everything the web prototype reads (data period, splits, all results, selected horizon and method)
- `outputs/short_term_results.csv`, `outputs/confusion_matrices.csv` — horizon × method × test set
- `outputs/time_of_day_results.csv` — time-of-day model vs historical averages
- `charts/` — macro-F1 and High-recall by horizon, confusion matrices, time-of-day comparison, data overview
- `models/short_term_rf_{10,30,60}min.joblib`, `models/time_of_day.joblib` — fitted on all readings after evaluation

## Scientific rules used here

- The short-term target is the congestion level observed **10/30/60 minutes after** the latest reading, matched by time (±5 min), so collection gaps cannot misalign it.
- Chronological splits only: train on earlier time, test on later time. The "usual ratio" and every other learned value are fitted on the training period only.
- Every model is shown next to two baselines: persistence (next level = current level) and the historical average.

See `PILOT_REPORT.md` for the results and their limits. These are **pilot-only** results.

## Web prototype (`web/`)

A Next.js + FastAPI interface for the Jeddah results (plus the NYC reference prototype). Its backend imports
`masar_features.py` from this folder and reads `models/`, `outputs/results.json` and the training snapshot
directly, so it always shows what `train_masar.py` produced. First time: run `web\SETUP_WINDOWS.bat`;
afterwards `web\RUN_PROJECT.bat`, then open http://localhost:3000. Details in `web/README.md`.
