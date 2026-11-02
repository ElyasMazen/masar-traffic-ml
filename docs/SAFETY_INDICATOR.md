# Masar — Safety Risk Indicator (Phase 1, offline)

**What it is:** a rule-based **safety risk indicator** built from speed patterns that the literature links to higher crash risk.
**What it is not:** accident prediction. There is no road-level crash data for these roads, so the indicator has **not been validated against crashes**. Every check below is a sanity or robustness check.

- **Code:** `masar_safety.py` (the indicator) and `safety_risk.py` (run + checks + charts)
- **Data:** raw ~10-minute TomTom readings, `data/snapshots/jeddah_tomtom_flow_20261024_145800.csv` (5–18 Oct 2026, 18,036 readings, 10 points). Incidents come from `../tomtomdata/jeddah_incidents.csv`, read only.
- **Outputs:**
  - `outputs/safety_risk.csv` (per reading)
  - `outputs/safety_validation.json` (all checks)
  - `charts/safety_*.png`
- **Congestion models:** not touched. The indicator only reads the snapshot.

```powershell
python safety_risk.py                       # defaults: 10-min step, 60-min window
python safety_risk.py --step 30             # same code on a 30-minute grid
python safety_risk.py --window 120          # 2-hour variability window
python safety_risk.py --with-incidents      # also computes an index with the optional incident component
```

## 1. Literature basis

The basis is the work cited in Report 1. Check the exact references against Report 1's reference list.

- **Choudhary et al. (2018)**, *Accident Analysis & Prevention*: speed **variability** on freeways is associated with higher crash risk.
- **Parsa et al. (2020)**, *Accident Analysis & Prevention* (XGBoost + SHAP for real-time crash detection): **speed differences between upstream and downstream** locations are among the strongest predictors.

Both studies use loop-detector data at 1–5-minute resolution and detectors a few hundred metres apart. Masar has 10-minute readings from TomTom segments that are kilometres long (§6). The indicator borrows the *factors*, not the models or their thresholds.

## 2. Method

### 2.1 Data and step
- Readings are loaded with `masar_features.load_data` after `set_step(step)`. At step 10 the grid keeps every raw reading (18,036 → 18,036); it only rounds each timestamp to its 10-minute slot. The actual time stays in `reading_ts`.
- The step is a setting (`SafetyConfig.step_min`, `--step`), so the same code runs on 30-minute data.
- Collection gaps use the existing rule: a previous reading counts only if it is at most one step + half a step older (`mf.lag_features`).

### 2.2 Components (per point and reading)

All four describe **change or instability**, not the congestion level itself.

| | Component | Formula | Notes |
|---|---|---|---|
| a | **Speed variability** | CV = std / mean of `current_speed_kmh` over the window (t − 60 min, t] for that point | Needs ≥ half the expected readings (≥ 3 at step 10). The default window is 60 min; 120 min is tested in §4c. |
| b | **Sudden drop** | max(0, ratio(t−1) − ratio(t)) | Drops only; increases count as 0. NaN across a collection gap. |
| c | **Neighbour difference** (approximate) | max over adjacent points on the same road of \|ratio(this) − ratio(neighbour)\| at the same grid time | See §2.3. |
| d | **Unusually slow** | max(0, usual ratio − ratio) | `usual` = point × hour × weekday/weekend mean from `mf.fit_usual` (all readings). |
| e | *Incident context (optional, off)* | Number of TomTom incidents active within 1 km | Not in the main index; see §4e. |

### 2.3 Point order and direction (component c)
- **Order:** the points of each road are ordered along the road's main axis (first principal component of their coordinates), north to south:
  - **Al Haramain:** 1 → 2 → 3 (10.7 km, 4.8 km apart)
  - **Al Madinah:** 1 → 2 → 3 → 4 (4.3, 3.3, 7.3 km)
  - **King Abdulaziz:** 3 → 2 → 1 (3.2, 4.0 km)
- **Direction: not known.** TomTom's flow segment can be either carriageway, and the CSV stores only the segment midpoint, not its orientation. So I can't say which point is upstream. The component therefore uses the **larger** difference to either neighbour.
- **Approximate.** The points are 3–11 km apart, and TomTom's matched segments are 2–20 km long and can overlap. Al Haramain 2's 19 km segment reaches past Al Haramain 3. Parsa et al. compare detectors a few hundred metres apart.

### 2.4 Index
- **Normalisation:** each component → 0–1 percentile rank over all readings. Ties take their **lowest** rank, so a component that is 0 scores exactly 0, not a mid rank. Free flow is the most common state, and 52–84% of readings are 0 on each component.
- **Index** = 100 × weighted mean of the available component scores, with **equal weights** (`SafetyConfig.weights`). It needs at least 3 of the 4 components; 50 readings (at the start of each point's series and right after collection gaps) have no index.
- **Main reason:** the component with the largest weighted contribution: "sudden drop", "high variability", "speed difference to neighbouring point" or "unusually slow for this hour". It is "none" when the index is 0.

### 2.5 Bands (two versions)
- **Relative:** top 10% of the index = **High**, next 20% = **Medium**, rest = Low. There is always 10% High, even in a calm week.
- **Absolute:** fixed cut-offs on the **raw** components. High if any component reaches its High cut-off, **or** two reach their Medium cut-off. Medium if one reaches its Medium cut-off.

| Component | Medium | High | Reasoning |
|---|---|---|---|
| Variability (CV) | 0.10 | 0.20 | ±10% of the mean speed within an hour is unsettled flow; ±20% (e.g. 50 ± 10 km/h) is stop-and-go |
| Sudden drop | 0.15 | 0.30 | Losing 15% of free-flow speed in one 10-min step (e.g. 90 → 76 km/h) is noticeable; 30% (90 → 63) is a sharp, shock-like slowdown |
| Neighbour difference | 0.30 | 0.50 | A 0.30 difference puts the two points in different congestion levels; 0.50 means free flow next to heavy congestion |
| Unusually slow | 0.15 | 0.30 | 15–30% of free-flow speed below what is normal for that hour and day type |

These are judgment values, not taken from the literature. With them, **22 of 120 point-days have no High at all**, vs 6 with the relative bands. Overall: High 1,173 (6.5%), Medium 3,170, Low 13,693.

## 3. Results overview (10-minute data)

- **Relative bands:** High 1,799, Medium 3,597, Low 12,590. The absolute and relative High sets agree on 478 readings.
- **Main reason of the relative High readings:** sudden drop 814, unusually slow 522, neighbour difference 245, high variability 218.
- **Index by hour (weekday mean):** low at night (1–15), rising sharply at 07:00 (44), dipping late morning, then 48–57 from 14:00 to 18:00. On weekends it is near 0 from 08:00 to 15:00 and rises from 17:00 (36–49 in the evening). See `charts/safety_risk_by_hour.png`.

## 4. Validation (sanity and robustness only, with no crash data)

### a) Behaviour
Congestion episodes are runs of `speed_ratio` < 0.85 lasting ≥ 30 min.

| Phase | Readings | Mean index | Share High |
|---|---|---|---|
| Night free flow (00–05 h) | 3,492 | **7.1** | 1.5% |
| Daytime free flow | 9,177 | 11.3 | 1.5% |
| **Onset** of an episode (± 1 reading) | 693 | 62.7 | **44.2%** |
| Inside an episode | 2,938 | 61.9 | 23.7% |
| **Peak** (within 0.05 of the episode minimum) | 835 | **71.4** | **44.0%** |
| **End** of an episode (± 1 reading) | 653 | 53.7 | 12.9% |
| Short dips (< 30 min, "other") | 248 | **77.5** | **63.3%** |

**Holds:**
- Low at night and in free flow.
- High at onsets and peaks, and highest for short, sudden dips.
- At every point, onset and peak are well above both free-flow phases.

**Does not hold, or holds only weakly:**
- **Episode ends** score *lower* than the inside of episodes. Recovery shows up as a speed *increase*, which the drop component ignores by design.
- The index stays high **throughout** congestion, not only at its edges (see the timeline chart).
- King Abdulaziz 3 scores 30 at night in free flow. That comes from small night-time speed variations on a road whose "free flow" varies.

### b) Overlap with congestion — **the index largely follows congestion**
- **Correlation with `speed_ratio`:** Spearman **−0.82** (Pearson −0.80). Each component alone: −0.61 to −0.77.
- **Where the relative High readings fall:** 32% in High congestion, 56% in Medium, 12% in Low (211 readings in free flow).

| Risk band (relative) \ congestion | Low | Medium | High |
|---|---|---|---|
| Low | 94.9% | 4.6% | 0.5% |
| Medium | 24.4% | 55.7% | 19.8% |
| High | 11.7% | 56.1% | 32.1% |

**Why:** TomTom caps speed at free flow. In free flow every component is 0: no variability is visible, there is no drop, the neighbour is also at 1.0, and nothing is below usual. Instability only becomes measurable once traffic slows, so an index of instability ends up close to "is it congested".

- **What helps:**
  - **Measuring risk relative to the congestion level** helps most. The index minus the mean index at the same speed-ratio band has Spearman **−0.09**, and 49% of its High readings fall in congestion (vs 88%).
  - Percentile ranks *within* each congestion level give −0.73.
- **What doesn't help:** using only the change components (variability + drop) still gives −0.80.

### c) Sensitivity
Overlap = the share of the baseline top 10% that stays in the top 10%.

| Variant | Overlap |
|---|---|
| Variability weight ×0.5 / ×1.5 | 97% / 98% |
| Sudden drop weight ×0.5 / ×1.5 | 91% / 99% |
| Neighbour weight ×0.5 / ×1.5 | 96% / 97% |
| Unusual weight ×0.5 / ×1.5 | 96% / 98% |
| Window 120 min instead of 60 | 93% |

The High set is robust to the weights and the window. Part of that robustness comes from the components being strongly correlated with one another.

### d) Isolation Forest cross-check
An Isolation Forest (300 trees, 10% contamination) on the same four components:
- on the percentile scores, it puts **29%** of our High readings in its top 10%;
- on the standardised raw components, **62%**.

So it agrees only partly. The forest looks for *rare combinations* of components, while the index rewards *large values*; on the percentile scores those are different readings.

### e) Weak proxy: TomTom incidents (**not accident validation**)
Incidents are mostly jams, i.e. congestion itself, and the data has **no accident** incidents. Coverage is 10–18 Oct, 12,206 readings.

| Incidents within 1 km | Readings affected | Mean index near / not near | Share High near / not near |
|---|---|---|---|
| Any (jam, closure, road works) | 11,512 (94%) | 24.4 / 14.0 | 9.1% / 3.3% |
| Short-lived jams (≤ 3 h) | 1,638 (13%) | 46.6 / 20.3 | 18.3% / 7.3% |
| Closures and road works | 5,736 (47%) | 21.5 / 25.9 | 8.7% / 8.7% |

- **Any incident:** an incident is "near" 94% of the time, because TomTom keeps some "Jam" incidents active for days. That gives no contrast.
- **Short jams:** the index is clearly higher, but within the same congestion level the gap almost disappears (Medium 59.7 vs 62.4, High 71.9 vs 71.8). It remains only in free flow (21.5 vs 9.7), which suggests the index rises as a jam forms.
- **Closures and road works:** no signal.

**Optional incident component:** it would affect 11,512 of the 12,206 covered readings. It is not in the main index, and with this definition it would add almost nothing.

### f) The 10 highest-risk moments
At most one per point within 2 hours. All are in High congestion and all are absolute High.

| # | Point | Time | Index | Main reason | Speed before → now → after (km/h) |
|---|---|---|---|---|---|
| 1 | King Abdulaziz 1 | Tue 13 Oct 13:30 | 100.0 | unusually slow | 37 → 15 → 28 |
| 2 | Al Madinah 2 | Sun 11 Oct 18:50 | 99.7 | sudden drop | 94 → 42 → 38 |
| 3 | Al Madinah 2 | Tue 13 Oct 07:20 | 99.6 | neighbour difference | 54 → 35 → 32 |
| 4 | King Abdulaziz 1 | Wed 14 Oct 14:20 | 99.6 | high variability | 45 → 25 → 37 |
| 5 | King Abdulaziz 1 | Fri 9 Oct 22:20 | 99.4 | unusually slow | 42 → 23 → 26 |
| 6 | Al Madinah 2 | Mon 12 Oct 07:30 | 99.3 | neighbour difference | 42 → 31 → 36 |
| 7 | King Abdulaziz 1 | Tue 6 Oct 14:10 | 99.3 | sudden drop | 46 → 26 → 25 |
| 8 | King Abdulaziz 1 | Mon 12 Oct 18:50 | 99.2 | high variability | 42 → 25 → 38 |
| 9 | Al Madinah 2 | Tue 6 Oct 09:50 | 99.0 | sudden drop | 94 → 49 → 94 |
| 10 | King Abdulaziz 1 | Wed 14 Oct 16:50 | 98.9 | high variability | 46 → 28 → 28 |

Clear examples:
- **#2:** 94 → 42 km/h in 10 min, a speed drop of more than half.
- **#9:** a 10-minute collapse from 94 to 49 km/h and back to 94.

All ten come from **King Abdulaziz 1 and Al Madinah 2**. These have the **shortest TomTom segments** (3.1 and 2.1 km), where one reading reacts most to local changes. The index therefore favours short segments, a bias to keep in mind.

### g) 30-minute grid vs 10-minute data
- **Data:** 6,056 readings instead of 18,036.
- **Lost High moments:** only **53%** of the 10-minute High moments are still High in their 30-minute slot.
- **Variability almost disappears:** a 1-hour window holds just 2 readings. Variability is the main reason for 3% of High readings at 30 min, vs 12% at 10 min.
- **Drops look bigger but are coarser:** a "sudden drop" is now measured over 30 min. The largest drop rises from 0.55 to 0.82, but a brief 10-minute collapse that recovers (like #9) is missed.
- **Overlap with congestion is slightly worse:** Spearman −0.83 vs −0.82.

## 5. Limitations
- **No crash validation.** There is no road-level crash data, so the indicator shows *patterns linked to risk in the literature*, not measured risk, and it must not be presented as accident prediction.
- **Time step:** 10 min now and 30 min after the collector restart, vs 1–5 min in the literature. Shock waves shorter than a step are invisible, and at 30 min variability is barely measurable.
- **Km-spaced points and long segments:** points 3–11 km apart and TomTom segments 2–20 km long, sometimes overlapping. The neighbour component is approximate, and travel direction is unknown.
- **Chosen weights:** the weights (equal) and the absolute cut-offs are judgment values; nothing learned them from crashes.
- **Speed capped at free flow:** TomTom reports at most the free-flow speed, so **speeding and high-speed variability are invisible**. As a result the index rises mainly when traffic slows and largely follows congestion (§4b).
- **Relative bands:** the relative bands always mark 10% of readings as High, even in a calm week. The absolute bands avoid that, but their cut-offs are chosen, not calibrated.
- **Short-segment bias:** points with short segments react more and dominate the highest scores.
- **In-sample "usual":** the usual ratio is fitted on the same 14 days it describes.

## 6. Future work
- **Validate, and learn the weights, once crash data is available** (e.g. police or Najm records with time and location), instead of using equal weights.
- **A supervised model following Parsa et al.:** XGBoost with SHAP explanations, using these same components as features and crashes as the target.
- **Finer time steps** (≤ 5 min) once the API budget allows, so short shock waves become visible.
- **Closer points:** pairs of points a few hundred metres apart around known bottlenecks, with known travel direction, for a real upstream/downstream difference.
- **Risk forecasting** from the existing speed forecasts (+30/+60 min), turning the forecast speeds into an expected risk.
