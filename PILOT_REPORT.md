# Masar — Pilot ML/Training Report

**Project:** Masar (مسار) — Smart Traffic Intelligence  
**Scope of this delivery:** Jeddah TomTom pilot data, next-observation congestion classification  
**Snapshot:** `jeddah_tomtom_flow 10.10(2).csv`  
**Important:** All accuracy values below are **pilot results**, not final project accuracy.

## 1. What was built

This delivery implements a leak-free, re-runnable pipeline for the current Masar traffic-flow snapshot. It mirrors the reference paper's methodological principles where they fit this dataset: chronological ordering, train-only preprocessing, held-out validation/test periods, temporal features, lagged traffic context, model comparison, and later extensibility to sequence models.

The task here is classification rather than the paper's `volumePerHour` regression. The target is the **next observed congestion level (~10 minutes ahead)** for each TomTom point:

- Low: `speed_ratio >= 0.85`
- Medium: `0.60 <= speed_ratio < 0.85`
- High: `speed_ratio < 0.60`

### Critical leakage decision

`congestion_level` is deterministically derived from the same row's `speed_ratio`. Therefore, using current `speed_ratio`, current speed, delay, or travel time from the target row would produce a misleadingly easy classification problem.

This pipeline **does not use contemporaneous target-row traffic measurements**. It predicts time `t` from measurements available at `t-1`, `t-2`, and `t-3`, plus known calendar/road information.

## 2. Data audit

- Rows: **5,810**
- Runs: **581**
- Points: **10**
- Roads: **3**
- Local coverage: **2026-10-05 20:29:44 → 2026-10-10 00:20:33**
- Null cells: **0**
- Duplicate `(timestamp_utc, point_id)` pairs: **0**
- Every recorded run contains all 10 points: **True**
- Approximate 10-minute schedule coverage: **96.83%**
- Gaps >15 minutes: **6**
- Maximum gap: **92.30 minutes**
- Speed-ratio formula mismatches: **0**
- Congestion-label threshold mismatches: **0**
- `road_closure=True` rows: **6**
- `speed_ratio == 1.0`: **63.13%** of rows
- Eligible model rows after continuity/lag filtering: **5,670**

### Items that need attention, not silent fixing

1. **Six >15-minute gaps exist.** Three are the large 5 Oct outage gaps, but there are also ~20-minute gaps on 7 Oct. These are listed in `outputs/gaps_over_15min.csv`.
2. **Six `road_closure=True` rows occur on King Abdulaziz Road point 3 while reported speeds remain 62–66 km/h.** This could be valid TomTom semantics (e.g., a closure flag affecting part of the returned segment) or an API-data inconsistency. The raw rows are preserved in `outputs/road_closure_rows_for_review.csv`; nothing was overwritten.
3. The target distribution is imbalanced: **Low 66.83% / Medium 23.46% / High 9.71%**.
4. Some points currently have almost no High samples. Thresholds should **not** be changed from only this short pilot. Re-check after at least ~2 weeks including more weekends, as planned.

## 3. Feature design

Known-at-forecast-time features:
- Road, point ID, FRC
- Time-of-day represented cyclically (`sin/cos`)
- Day-of-week represented cyclically
- Saudi weekend flag

Past-only dynamic features:
- `speed_ratio` lags 1–3
- `delay_s` lags 1–3
- current-speed lags 1–3
- previous travel/free-flow values
- previous confidence and closure flag
- previous congestion level
- rolling mean/std over the last 3 observations

A training sample is retained only when the target interval and the previous two intervals are each no more than 15 minutes. This avoids pretending an outage gap is a normal 10-minute step.

## 4. Chronological split

The split is performed by **unique timestamps**, so the 10 points from one run never leak across train/validation/test.

- Train: 72%
- Validation: 8%
- Test: 20%

Current pilot test period is mostly Friday and has fewer High examples than development data. That is a real distribution shift and is one reason these results must remain labelled **pilot**.

## 5. Models

Three fixed, conservative ML baselines were trained without Bayesian over-tuning:

1. Logistic Regression
2. Decision Tree
3. Random Forest

A non-ML **Persistence** baseline was also evaluated: predict that the next level stays equal to the latest observed level.

Model family selection uses **validation Macro-F1**, not the test set.

**Selected ML model: Random Forest.**

## 6. Pilot results

### Validation
Random Forest:
- Accuracy: **0.9200**
- Balanced accuracy: **0.8786**
- Macro-F1: **0.8617**

### Untouched chronological test
Random Forest:
- Accuracy: **0.9570**
- Balanced accuracy: **0.8796**
- Macro-F1: **0.8873**
- Low F1: **0.9803**
- Medium F1: **0.8874**
- High F1: **0.7941**

Persistence baseline on the same test:
- Accuracy: **0.9588**
- Balanced accuracy: **0.8803**
- Macro-F1: **0.8890**

The persistence baseline is slightly stronger overall than Random Forest in this very short pilot. That is plausible because traffic states are highly persistent across adjacent ~10-minute intervals. It creates a meaningful baseline that LSTM/GRU should beat later; it should not be hidden.

## 7. Interpretation

The current model is mostly learning **short-term traffic persistence and recent speed-ratio/delay history**. Random Forest feature importance confirms that the strongest predictors are recent `speed_ratio`, its 3-step rolling mean, recent delay, and the previous congestion state.

This is exactly why the current accuracy should **not** be presented as final. The dataset covers only a few days, the test period has a different class mix, and the same recurring road/point patterns have not yet been observed over enough weeks.

## 8. Saved models

- `models/random_forest_evaluated_dev80.joblib`  
  Trained on train+validation after model selection. The 20% test remained untouched for the reported score.

- `models/random_forest_pilot_fullfit.joblib`  
  Separate demo/prototype model fit on all eligible pilot rows **after evaluation**. Use this one for the simple interface/demo. Its full-fit status must not be confused with the reported held-out test result.

## 9. Recommended next stages

When more data arrives:

1. Re-run this exact pipeline weekly on downloaded snapshots.
2. After ~2 weeks, review congestion thresholds and per-point class support.
3. Add a simple **historical/time-only baseline** for forecasts that do not have recent live observations.
4. When enough repeated daily/weekly patterns exist, perform limited validation-only tuning.
5. Add LSTM and GRU using sliding windows adapted to 10-minute data. A 10-step window in Masar means about **100 minutes**, unlike the 10 minutes represented by 10 one-minute steps in the reference study.
6. Compare ML, persistence, LSTM, and GRU on the same chronological test framework.
7. Add SHAP/feature-importance analysis and regime analysis.
8. Later, add road-safety scoring as a separate target/task rather than mixing it into congestion labels.

## 10. Files to use

- `train_masar.py` — re-train on any downloaded snapshot
- `predict_next.py` — generate next-step point forecasts
- `masar_features.py` — shared feature engineering
- `outputs/test_metrics.csv` — final pilot comparison table
- `outputs/latest_next_step_forecast.csv` — demo forecasts from the latest snapshot
- `outputs/random_forest_feature_importance.csv`
- `outputs/confusion_matrix_random_forest.csv`
- `charts/` — presentation-ready figures
- `models/` — evaluated and demo models

## 11. Reference methodology

Sara Atef (2026), *Comparative Evaluation of Traffic Load Prediction Models for Intelligent Transportation Systems Using High-Resolution Urban Data*, Smart Cities, 9(4), 56. DOI: 10.3390/smartcities9040056.

Masar adapts the methodological principles but does **not** claim an identical experiment: our current data are 10-minute multi-segment observations and the current target is categorical congestion level, while the paper predicts traffic volume from one-minute data.
