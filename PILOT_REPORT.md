# Masar — Pilot ML/Training Report

**Project:** Masar (مسار) — Smart Traffic Intelligence
**Scope:** Jeddah TomTom pilot data, 10 points on 3 roads, readings every ~10 minutes
**Data used:** 11,150 readings, Mon 5 Oct 2026 20:29 → Tue 13 Oct 2026 17:20 (9 calendar days: 7 weekdays — Mon 5, Tue 6, Wed 7, Thu 8, Sun 11, Mon 12, Tue 13 — and 2 weekend days, Fri 9 and Sat 10; Mon 5 and Tue 13 are partial)
**Snapshot:** `data/snapshots/jeddah_tomtom_flow_20261013_172552.csv` (downloaded from the raw GitHub CSV; re-run with `python train_masar.py --csv <that file>`)
**Important:** every number below is a **pilot result**, not final project accuracy.

## 1. What changed since the previous run

The previous run (5–10 Oct, 6,990 readings) was tested on a single held-out weekday (Thu 8 Oct). Its 6,990 readings had no Sunday. This run:

1. Uses 4,160 more readings (now 11,150), including the first Sunday (11 Oct) and two more weekdays.
2. Replaces the single-Thursday test with a **multi-day chronological holdout**. It covers Sat 10 Oct 00:00 → Tue 13 Oct 17:20, and the models train on Mon 5 → Fri 9 (one weekend day). A full held-out week is not possible yet, because that would leave only one evening to train on.
3. Reports the weekday and weekend parts of that test separately.
4. Explicitly separates the **evaluated** models from the **full-data** models used by the app (section 2).

The features, the baselines, the Random Forest settings and the Low/Medium/High thresholds (0.85 / 0.60) are unchanged.

## 2. Evaluated models vs the models the app uses

| | Trained on | Scored? | Used for |
|---|---|---|---|
| **Evaluated models** | Mon 5 Oct 20:29 → Fri 9 Oct 23:50 (5,780 readings) | Yes — every metric in this report, on the held-out test only | Measuring accuracy |
| **Full-data models** | All 11,150 readings, 5 Oct → 13 Oct 17:20 (train + test) | No — refitted with the same settings after evaluation | Every prediction in the web app and `predict_next.py` |

Refitting on all data is standard practice. The full-data models have seen more data, including the latest days, but they have no held-out score of their own. The held-out numbers are the best available estimate of how they perform. The web app labels both models the same way.

## 3. Test sets (chronological only)

| Test set | Train on | Test on | Day types in test |
|---|---|---|---|
| **Main: last 4 days** | 5 Oct 20:29 → 9 Oct 23:50 | Sat 10 Oct 00:00 → Tue 13 Oct 17:20 (5,370 readings) | weekday (Sun, Mon, Tue) **and** weekend (Sat) |
| Secondary: latest 20% of time | 5 Oct 20:29 → 12 Oct 04:10 | 12 Oct 04:20 → 13 Oct 17:20 (2,230 readings) | weekday only |

The secondary test lies inside the main test period, with more training days, so it is not an independent second test. The horizon and the time-of-day method are chosen on the main test.

## 4. Short-term results (+10 / +30 / +60 min)

Macro-F1 / accuracy / High recall (recall of the "High" congestion class). **Bold** = best macro-F1 in each group.

**Main test — all rows (Sat 10 → Tue 13)**

| Horizon | Method | Macro-F1 | Acc | High recall |
|---|---|---|---|---|
| +10 min | Persistence | **0.873** | 0.931 | 0.86 |
| +10 min | Historical average | 0.665 | 0.813 | 0.58 |
| +10 min | Random Forest | 0.861 | 0.919 | 0.89 |
| +30 min | Persistence | **0.786** | 0.885 | 0.75 |
| +30 min | Historical average | 0.665 | 0.813 | 0.58 |
| +30 min | Random Forest | 0.776 | 0.869 | 0.85 |
| +60 min | Persistence | 0.680 | 0.828 | 0.60 |
| +60 min | Historical average | 0.666 | 0.814 | 0.59 |
| +60 min | Random Forest | **0.716** | 0.828 | 0.77 |

**Main test, split by day type — macro-F1 (High recall in brackets)**

| Horizon | Method | Weekdays: Sun 11, Mon 12, Tue 13 | Weekend: Sat 10 |
|---|---|---|---|
| +10 min | Persistence | **0.876** (0.87) | 0.764 (0.53) |
| +10 min | Random Forest | 0.862 (0.90) | **0.766** (0.67) |
| +30 min | Persistence | **0.780** (0.75) | **0.711** (0.50) |
| +30 min | Random Forest | 0.771 (0.86) | 0.686 (0.58) |
| +60 min | Persistence | 0.669 (0.61) | **0.624** (0.44) |
| +60 min | Historical average | 0.668 (0.60) | 0.530 (0.22) |
| +60 min | Random Forest | **0.713** (0.77) | 0.596 (0.33) |

High readings: about 400 in the weekday part, only **9–15 in the weekend part**, so the weekend recalls rest on a handful of readings.

**Secondary test (latest 20%, weekdays):** +10 min RF 0.872 vs persistence 0.872; +30 min 0.783 vs 0.770; +60 min **0.747 vs 0.650**, with High recall 0.90 vs 0.58.

**Interpretation.** As before, at +10 and +30 minutes the Random Forest does **not** beat persistence: traffic barely changes in 10–30 minutes. At +60 minutes it is ahead on weekdays (0.713 vs 0.669 macro-F1) and catches clearly more High-congestion readings (77% vs 61%). On the single test Saturday it is behind persistence (0.596 vs 0.624), because it trained on only one weekend day. The weekday result is better supported than last time (three test weekdays instead of one), but it is still **indicative only**.

The web app shows the +60 min forecast in its own "Next hour (from latest reading)" section, next to persistence.

Figures: `charts/short_term_macro_f1.png`, `charts/short_term_high_recall.png`, `charts/confusion_holdout_<horizon>min.png`, `charts/confusion_latest_<horizon>min.png`.

## 5. Time-of-day prediction (cards, map, 12-hour chart)

Predicts the usual speed for any point, date and hour, with no recent readings needed. Speed error = mean absolute error of the predicted speed. Level acc. = share of readings with the right level.

| Method | Main test, all | Weekdays only | Weekend only | Latest 20% (weekdays) |
|---|---|---|---|---|
| Gradient-boosting time model | **±4.4 km/h**, 83% | **±4.4**, 83% | ±4.2, 85% | **±3.6**, 85% |
| Historical average, point × hour | ±6.2 km/h, 74% | ±5.6, 74% | ±8.0, 72% | ±5.5, 76% |
| Historical average, point × hour × weekday/weekend | ±4.7 km/h, 81% | ±4.9, 80% | **±4.2**, 85% | ±4.2, 83% |

The time model has the lowest error on the main test, so the app keeps it. On the weekend day it ties with the weekday/weekend average. Mixing weekdays and weekends (point × hour) almost doubles the weekend error, as in the previous run. A day type with no data at an hour falls back to the weekday pattern, never the weekend one. Note the weekend High recall of the time model is very low (1 of 16 High readings on Saturday): it predicts the usual, light weekend traffic.

## 6. Labels and data audit

- **Labels (all readings, thresholds unchanged):** Low 7,936 (71.2%) / Medium 2,235 (20.0%) / High 979 (8.8%). The previous run had 69.9 / 21.9 / 8.2%.
- **High readings per point:**

  | Point | High | Point | High |
  |---|---|---|---|
  | Al Haramain Rd 1 | 0 | Al Madinah Rd 3 | 188 |
  | Al Haramain Rd 2 | 256 | Al Madinah Rd 4 | 144 |
  | Al Haramain Rd 3 | 176 | King Abdulaziz Rd 1 | 155 |
  | Al Madinah Rd 1 | 0 | King Abdulaziz Rd 2 | 7 |
  | Al Madinah Rd 2 | 28 | King Abdulaziz Rd 3 | 25 |

  Two points have never been High, and three have fewer than 30 High readings. The High-class results come almost entirely from five points. See `outputs/label_counts_by_point.csv`.
- 1,115 runs, all 10 points per run. 6 gaps longer than 15 minutes, all on 5–7 Oct (max 92 min, on 5 Oct); none since then. See `outputs/gaps_over_15min.csv`.
- 6 `road_closure=True` rows on King Abdulaziz Rd point 3 while speeds stay normal. They are unchanged and kept for review in `outputs/road_closure_rows_for_review.csv`.

## 7. Limits

- **Two weekend days in total, one of them in the test.** Weekend results rest on one Saturday and about 10 High readings, so differences there are noise-level.
- **Three test weekdays, one Sunday.** This is better than one test day, but differences of a few hundredths of F1 are still within noise.
- **No independent second test.** The secondary test overlaps the main one.
- The probabilities shown in the app are the forest's average class probabilities, **not calibrated confidence**.

## 8. Next steps

1. Re-run `train_masar.py` weekly. Once two full weeks exist, hold out the last full week (weekdays + a weekend) instead.
2. Re-check whether the +60 min weekday edge holds with more weeks, and whether more weekend data closes the weekend gap.
3. Re-add day-of-week detail to the time model once each weekday has been seen several times.
4. Review the 0.85 / 0.60 thresholds with more weeks of data, especially for the points that are almost never High.
5. Try LSTM/GRU only after the Random Forest clearly beats persistence on several held-out weeks.

## 9. Files

- `masar_features.py` — the single data/feature/baseline module (`load_data` is the only entry point for data); the web app in `web/` imports it directly
- `train_masar.py` — full evaluation, charts and final (full-data) models
- `predict_next.py --horizon {10,30,60}` — forecast from the latest readings with the full-data model
- `outputs/`, `charts/`, `models/` — see README; `outputs/results.json` → `models` records which period each kind of model was trained on

## 10. Reference methodology

Sara Atef (2026), *Comparative Evaluation of Traffic Load Prediction Models for Intelligent Transportation Systems Using High-Resolution Urban Data*, Smart Cities, 9(4), 56. DOI: 10.3390/smartcities9040056. Masar adapts its methodological principles (chronological evaluation, train-only preprocessing, baselines) but is not the same experiment: it uses 10-minute multi-point data and a categorical congestion target.
