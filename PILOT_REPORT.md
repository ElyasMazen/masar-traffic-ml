# Masar — Pilot ML/Training Report

**Project:** Masar (مسار) — Smart Traffic Intelligence
**Scope:** Jeddah TomTom pilot data, 10 points on 3 roads
**Data used:** 18,036 readings collected every ~10 minutes, Mon 5 Oct 2026 20:29 → Sun 18 Oct 2026 12:20. That is 14 calendar days: 10 weekdays and 4 weekend days (Fri 9, Sat 10, Fri 16, Sat 17). Mon 5 and Sun 18 are partial.
**Modelling grid:** 30 minutes (default). The readings are resampled to 6,056 grid readings; see section 2.
**Snapshot:** `data/snapshots/jeddah_tomtom_flow_20261024_145800.csv` (re-run with `python train_masar.py --csv <that file>`)
**Important:** every number below is a **pilot result**, not final project accuracy.

## 1. What changed since the previous run

1. **Collection stopped on 18 Oct.** TomTom's free tier is 20,000 flow requests per **month**, not per day, so the collector ran out of quota. When it restarts it will run every **30 minutes** instead of every 10. The data until 18 Oct is at a 10-minute step; the data after the restart will be at a 30-minute step.
2. **The pipeline now has a configurable grid step.** It is set by `STEP_MIN` in `masar_features.py`, `--step` on `train_masar.py`, and defaults to 30. Every reading is resampled to that step per point, so 10- and 30-minute data can be mixed. Everything else follows the step:
   - lags and the gap limit (lag k = k steps)
   - the target tolerance (± half a step)
   - the horizons (step 30 → +30/+60 min; step 10 → +10/+30/+60 min)
   - the backend's hour averaging (2 slots per hour at step 30)
3. **New holdout rule.** The test starts at midnight of the most recent Friday such that the test holds that full Fri+Sat weekend **and** training holds at least one full Fri+Sat weekend of its own. The old rule picked Sat 10 Oct again, which left training with only 5–9 Oct and sent the second weekend entirely to the test. If no Friday qualifies, the old rule is used and `results.json` says so (`split_rule.fallback`).
4. **The "latest 20%" test is skipped this time.** It would start at the same midnight as the main test, so it would repeat the same test and count it twice. The reason is recorded in `results.json` (`split_rule.latest_split`).

## 2. Resampling to the 30-minute grid

Each reading is assigned to the **nearest** grid time (e.g. 12:00, 12:30). If several readings fall on the same grid time, the one closest to it is kept, and its real collection time stays in `reading_ts`.

**Why nearest and not a per-bin mean:** after the restart the collector delivers one instantaneous reading per 30-minute slot. A mean of three 10-minute readings would be smoother (less variable) than what the models will see from then on, so the old and new data would not be comparable.

On this snapshot, 18,036 readings become **6,056 grid readings** (11,980 merged). The newest reading (18 Oct 12:20:28) falls in the 12:30 slot.

## 3. Evaluated models vs the models the app uses

| | Trained on | Scored? | Used for |
|---|---|---|---|
| **Evaluated models** | Mon 5 Oct 20:30 → Thu 15 Oct 23:30 (4,840 grid readings; includes the weekend of 9–10 Oct) | Yes — every metric in this report, on the held-out test only | Measuring accuracy |
| **Full-data models** | All 6,056 grid readings, 5 Oct → 18 Oct 12:30 | No — refitted with the same settings after evaluation | Every prediction in the web app and `predict_next.py` |

## 4. Test set

| Test | Train | Test | Day types |
|---|---|---|---|
| **Last 3 days** (main) | 5 Oct 20:30 → 15 Oct 23:30 | **Fri 16 Oct 00:00 → Sun 18 Oct 12:30** (1,216 grid readings) | weekend (Fri 16, Sat 17) + weekday (Sun 18, morning only) |

Rule check: the test holds the full weekend of 16–17 Oct, and training holds the full weekend of 9–10 Oct, so no fallback was needed.

## 5. Short-term results (+30 / +60 min, step 30)

Macro-F1 / High recall (share of real High readings that were predicted High). **Bold** = best macro-F1 in each group.

| Horizon | Method | All (n=1,206 / 1,196) | Weekend: Fri 16, Sat 17 | Weekday: Sun 18 (morning) |
|---|---|---|---|---|
| +30 min | Persistence | 0.827 / 0.76 | **0.792** / 0.65 | 0.836 / 0.92 |
| +30 min | Historical average | 0.650 / 0.21 | 0.548 / 0.00 | 0.664 / 0.50 |
| +30 min | Random Forest | **0.832** / 0.86 | 0.791 / 0.76 | **0.885** / 1.00 |
| +60 min | Persistence | 0.729 / 0.61 | 0.672 / 0.44 | 0.770 / 0.83 |
| +60 min | Historical average | 0.653 / 0.21 | 0.548 / 0.00 | 0.664 / 0.50 |
| +60 min | Random Forest | **0.769** / 0.75 | **0.731** / 0.63 | **0.800** / 0.92 |

High readings in the test: **29 at +30 min and 28 at +60 min** in total; 16–17 on the weekend days and 12 on Sunday morning.

**Interpretation.**
- At **+30 min** the Random Forest is level with persistence (0.832 vs 0.827, below the 0.01 noise margin).
- At **+60 min** it is ahead (0.769 vs 0.729) and catches more High readings (75% vs 61%). Unlike the previous run, this time it is also ahead on the weekend days.
- The app therefore keeps +60 min.
- With fewer than 30 High readings and only one Sunday morning, this is **indicative only**.

## 6. Time-of-day prediction (cards, map, 12-hour chart)

| Method | All: speed error, level acc. | Weekend (Fri, Sat) | Weekday (Sun morning) |
|---|---|---|---|
| Gradient-boosting time model | ±2.71 km/h, 91% | ±2.83, 92% | **±2.26**, 90% |
| Historical average, point × hour | ±7.27 km/h, 74% | ±8.27, 71% | ±3.53, 87% |
| Historical average, point × hour × weekday/weekend | **±2.70 km/h**, 90% | **±2.65**, 90% | ±2.89, 90% |

The selection rule (lowest speed error on the main test) now picks the **point × hour × weekday/weekend average**, by 0.012 km/h. That difference is far below anything meaningful: the two methods are tied. The app currently shows the average because of how the rule works, not because it is clearly better. Mixing weekdays and weekends (point × hour) is again much worse on weekend days (±8.3 km/h).

## 7. Step 10 vs step 30 (same snapshot, same split)

Both runs train on 5–15 Oct and test on 16–18 Oct. Only the grid step differs. The step-10 outputs were not committed.

| Metric (main test, all rows) | Step 10 | Step 30 |
|---|---|---|
| Grid readings (train / test) | 14,410 / 3,626 | 4,840 / 1,216 |
| +10 min: RF / persistence macro-F1 | 0.886 / **0.895** | — |
| +10 min: RF / persistence High recall | 0.92 / 0.84 | — |
| +30 min: RF / persistence macro-F1 | **0.820** / 0.808 | **0.832** / 0.827 |
| +30 min: RF / persistence High recall | 0.83 / 0.71 | 0.86 / 0.76 |
| +60 min: RF / persistence macro-F1 | **0.790** / 0.729 | **0.769** / 0.729 |
| +60 min: RF / persistence High recall | 0.72 / 0.58 | 0.75 / 0.61 |
| High readings in test at +60 min | 81 | 28 |
| Time-of-day model MAE | ±2.60 km/h | ±2.71 km/h |
| Weekday/weekend average MAE | ±2.67 km/h | ±2.70 km/h |
| Selected horizon / time-of-day method | +60 / time model | +60 / weekday/weekend average |

**What changes.**
- At step 30 the RF loses its +10 min horizon and has a third of the training rows. At +60 min its margin over persistence falls from 0.061 to 0.040, and the speed error rises slightly.
- Persistence at +60 min is the same at both steps (0.729): it does not learn anything.
- The overall picture is unchanged: the RF only clearly beats persistence at +60 min.

## 8. Labels and data audit (30-minute grid)

- **Labels (thresholds unchanged, 0.85 / 0.60):** Low 4,399 (72.6%), Medium 1,194 (19.7%), High 463 (7.7%). On all 18,036 raw readings: Low 13,080, Medium 3,596, High 1,360.
- **High per point (grid readings):**

  | Point | High | Point | High |
  |---|---|---|---|
  | Al Haramain Rd 1 | 1 | Al Madinah Rd 3 | 83 |
  | Al Haramain Rd 2 | 130 | Al Madinah Rd 4 | 62 |
  | Al Haramain Rd 3 | 78 | King Abdulaziz Rd 1 | 78 |
  | Al Madinah Rd 1 | 0 | King Abdulaziz Rd 2 | 3 |
  | Al Madinah Rd 2 | 9 | King Abdulaziz Rd 3 | 19 |

- **Gaps:** gaps longer than 1.5 grid steps (45 min) are listed in `outputs/collection_gaps.csv`. There are 2, both on the evening of 5 Oct (90 and 60 min). Shorter 10-minute gaps disappear on the 30-minute grid.
- **End of collection:** collection stopped on 18 Oct at 12:20. That is the end of the data, not a gap.
- **Data checks:** 0 rows were dropped by the validation checks.
- **Road closures:** 3 `road_closure=True` grid readings remain, kept for review.

## 9. Limits

- **The test is mostly weekend.** Only Sunday morning represents weekdays, so the weekday numbers rest on one half-day and 12 High readings.
- **Two weekends in total**, one in training and one in the test.
- **Time-of-day selection is a tie** (0.012 km/h). A small change in data can flip it.
- **No 30-minute data from the collector exists yet.** The 30-minute grid is built from 10-minute data. Once the collector restarts, its readings may drift within the slot (GitHub Actions start times vary). The nearest-reading rule handles that, but it has not been tested on real 30-minute data.
- The probabilities shown in the app are the forest's average class votes, **not calibrated confidence**.

## 10. Next steps

1. When collection restarts at 30 minutes, check that the readings land near :00/:30, and re-run `train_masar.py` weekly.
2. Once two more full weeks exist, hold out a full week (weekdays + a weekend).
3. If the time-of-day methods stay tied, prefer the simpler average, or add a tie rule (e.g. keep the current method unless the new one is better by > 0.1 km/h).
4. Review the 0.85 / 0.60 thresholds with more data, especially for the points that are almost never High.

## 11. Files

- `masar_features.py` — the single data/feature/baseline module: `load_data`, `prepare` (validation + resampling), `set_step`. The web app in `web/` imports it directly.
- `train_masar.py [--step 30] [--csv …]` — full evaluation, charts and final (full-data) models
- `predict_next.py --horizon {30,60}` — forecast from the latest readings with the full-data model; it uses the step the model was trained with
- `outputs/`, `charts/`, `models/` — see README. `outputs/results.json → step` records the grid step, horizons and resampling counts.

## 12. Reference methodology

Sara Atef (2026), *Comparative Evaluation of Traffic Load Prediction Models for Intelligent Transportation Systems Using High-Resolution Urban Data*, Smart Cities, 9(4), 56. DOI: 10.3390/smartcities9040056. Masar adapts its methodological principles (chronological evaluation, train-only preprocessing, baselines) but is not the same experiment: it uses 30-minute multi-point data and a categorical congestion target.
