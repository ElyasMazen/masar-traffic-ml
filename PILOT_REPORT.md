# Masar — Pilot ML/Training Report

**Project:** Masar (مسار) — Smart Traffic Intelligence
**Scope:** Jeddah TomTom pilot data, 10 points on 3 roads, readings every ~10 minutes
**Data used:** 6,990 readings, Mon 5 Oct 2026 20:29 → Sat 10 Oct 2026 20:00 (6 calendar days; Monday covers only its last 3.5 hours)
**Snapshot:** `data/snapshots/jeddah_tomtom_flow_20261010_200107.csv` (downloaded from the raw GitHub CSV; re-run with `python train_masar.py --csv <that file>`)
**Important:** every number below is a **pilot result**, not final project accuracy.

## 1. What changed in this version

The previous pilot reported a 10-minute Random Forest with 96% accuracy and macro-F1 0.888. The persistence baseline ("next level = current level") scored 0.889 on the same test, so the model added nothing, and at a 10-minute horizon a 96% headline is misleading. This version:

1. Forecasts **+10, +30 and +60 minutes** ahead, with the target matched by time (the reading within ±5 min of t + horizon), so gaps cannot misalign it.
2. Uses current and lagged speed ratio, speed and delay (t … t−6), rolling mean/std, the change since the previous reading, hour (sin/cos), day of week, Saudi weekend, point, free-flow speed, and the "usual ratio for this point and hour" (weekday and weekend kept apart), **fitted on the training period only**.
3. Always compares the Random Forest (`class_weight="balanced"`) with **persistence** and the **historical average** on the same test rows.
4. Re-evaluates the time-of-day prediction behind the web app's cards, map and 12-hour chart against historical averages.

## 2. Test sets (chronological only)

| Test set | Train on | Test on | Day type |
|---|---|---|---|
| Latest 20% of time | 5 Oct 20:29 → 9 Oct 20:40 | 9 Oct 20:50 → 10 Oct 20:00 | **weekend only** (Fri evening + Sat) |
| Last full weekday | 5 Oct 20:29 → 7 Oct 23:50 | Thu 8 Oct, full day | **weekday only** |

The latest days of the data are a weekend, and Jeddah weekend afternoons are almost free-flowing (speed ratio 0.96–1.00 at 14:00–17:00, against 0.64–0.77 on weekdays). A test on the latest days alone therefore says little about weekday rush hours. The second test holds out the last full weekday instead. Neither test set mixes weekdays and weekends, so each row below is already a weekday-only or weekend-only result. There is **no Sunday** in the data yet.

## 3. Short-term results

Macro-F1 / accuracy / High recall (recall of the "High" congestion class). Bold = best macro-F1 in that row group.

| Horizon | Method | Thursday 8 Oct (weekday) — F1 | Acc | High recall | Latest (weekend) — F1 | Acc | High recall |
|---|---|---|---|---|---|---|---|
| +10 min | Persistence | **0.891** | 0.916 | 0.86 | **0.845** | 0.941 | 0.72 |
| +10 min | Historical average | 0.683 | 0.765 | 0.57 | 0.553 | 0.871 | 0.04 |
| +10 min | Random Forest | 0.885 | 0.911 | 0.87 | 0.843 | 0.938 | 0.76 |
| +30 min | Persistence | **0.815** | 0.862 | 0.75 | **0.788** | 0.915 | 0.65 |
| +30 min | Historical average | 0.685 | 0.766 | 0.57 | 0.556 | 0.877 | 0.05 |
| +30 min | Random Forest | 0.781 | 0.839 | 0.65 | 0.774 | 0.914 | 0.72 |
| +60 min | Persistence | 0.744 | 0.804 | 0.66 | 0.701 | 0.875 | 0.57 |
| +60 min | Historical average | 0.686 | 0.765 | 0.57 | 0.563 | 0.884 | 0.05 |
| +60 min | Random Forest | **0.757** | 0.817 | 0.61 | **0.728** | 0.905 | 0.57 |

High support: 197 High readings in the Thursday test, 37–46 in the weekend test.

**What this means.** At +10 and +30 minutes the Random Forest does **not** beat persistence on either test. Traffic states are very persistent over 10–30 minutes, and four days of training cannot teach the model when they change. At +60 minutes it is ahead on both tests, but only slightly (+0.013 and +0.027 macro-F1). On Thursday it gets that edge on the Medium class while catching **fewer** High readings than persistence (61% vs 66%). With a single held-out weekday, this is **indicative only**. It is not evidence that the model is better.

The web app shows the +60 min forecast, with persistence next to it and the comparison sentence above. Figures: `charts/short_term_macro_f1.png`, `charts/short_term_high_recall.png`, `charts/confusion_<test>_<horizon>min.png`.

## 4. Time-of-day prediction (cards, map, 12-hour chart)

Predicts the usual speed for any point, date and hour, with no recent readings needed.

| Method | Thursday (weekday): speed error | Level acc | Latest (weekend): speed error | Level acc |
|---|---|---|---|---|
| Gradient-boosting time model | **±4.9 km/h** | 79% | **±2.8 km/h** | 89% |
| Historical average, point × hour | ±5.5 km/h | 77% | ±7.8 km/h | 73% |
| Historical average, point × hour × weekday/weekend | ±5.5 km/h | 77% | ±3.1 km/h | 87% |

The time model has the lowest error on both tests, so the app keeps it (selection rule: lowest speed error on the weekday test). The two averages coincide on Thursday because its training days (Mon–Wed) include no weekend. On the weekend test, keeping weekdays and weekends apart cuts the error from ±7.8 to ±3.1 km/h.

The previous app version was worse than its own baseline (±13.3 vs ±11.8 km/h). That version used day-of-week features learned from one example per day, and averages that mixed weekend days into weekday hours. That mixing, plus a default date that fell on a Saturday, also produced the wrong-looking 12-hour chart. Now a weekday forecast for King Abdulaziz Rd point 1 tracks the observed weekday average: it falls from 0.84 at 13:00 to 0.38 at 17:00 and recovers after 19:00, and the 14:00–19:00 congestion appears where the data shows it. A day with no data yet, such as Sunday, uses the **weekday** pattern, never the weekend one.

## 5. Data audit

- 699 runs, all 10 points per run; 6 gaps longer than 15 minutes (max 92 min, on 5 Oct). See `outputs/gaps_over_15min.csv`.
- 6 `road_closure=True` rows on King Abdulaziz Rd point 3 while speeds stay normal. They are kept as-is for review in `outputs/road_closure_rows_for_review.csv`.
- Labels: Low 69.9% / Medium 21.9% / High 8.2%.

## 6. Limits that cannot be fixed with the current data

- **One weekday test day, no Sunday, two weekend days.** Every comparison above rests on very little data; differences of a few hundredths of F1 are within noise.
- **Weekday and weekend can't be compared within one test**, because each test set contains only one day type.
- Probabilities shown in the app are the forest's average class probabilities, **not calibrated confidence**.

## 7. Next steps

1. Re-run `train_masar.py` weekly. After 2–3 weeks, hold out a full week (weekdays and a weekend) instead of one day.
2. Re-add day-of-week detail to the time model once each weekday has been seen several times.
3. Review the 0.85 / 0.60 thresholds with more weeks of data.
4. Try LSTM/GRU only after the Random Forest clearly beats persistence; until then, persistence is the bar to clear.

## 8. Files

- `masar_features.py` — the single data/feature/baseline module (`load_data` is the only entry point for data); the web app in `web/` imports it directly
- `train_masar.py` — full evaluation, charts and final models
- `predict_next.py --horizon {10,30,60}` — forecast from the latest readings
- `outputs/`, `charts/`, `models/` — see README

## 9. Reference methodology

Sara Atef (2026), *Comparative Evaluation of Traffic Load Prediction Models for Intelligent Transportation Systems Using High-Resolution Urban Data*, Smart Cities, 9(4), 56. DOI: 10.3390/smartcities9040056. Masar adapts its methodological principles (chronological evaluation, train-only preprocessing, baselines) but is not the same experiment: it uses 10-minute multi-point data and a categorical congestion target.
