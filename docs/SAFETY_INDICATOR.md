# Masar — Safety Risk Indicator

**What it is:** a rule-based **safety risk indicator**. It flags moments when traffic at a point is **more unstable than usual for its traffic level**, using speed patterns that the literature links to higher crash risk.

**What it is not:** accident prediction. There is no road-level crash data for these roads, so the indicator has **not been validated against crashes**. Every check in this document is a sanity or robustness check.

- **Code:** `masar_safety.py` (the indicator) and `safety_risk.py` (run, checks, charts)
- **Data:** raw ~10-minute TomTom readings, `data/snapshots/jeddah_tomtom_flow_20261024_145800.csv` (5–18 Oct 2026, 18,036 readings, 10 points). Incidents come from `../tomtomdata/jeddah_incidents.csv`, read only.
- **Outputs:**
  - `outputs/safety_risk.csv` (per reading)
  - `outputs/safety_usual.csv` (per point × day type × hour, used by the web app)
  - `outputs/safety_validation.json` (all checks)
  - `charts/safety_*.png`
- **Congestion models:** not touched. The indicator only reads the snapshot.

```powershell
python safety_risk.py                       # final definition: 10-min data, 60-min window
python safety_risk.py --with-incidents      # also computes the optional incident component (not in the index)
```

---

## 1. Literature basis

The basis is the work cited in Report 1. Check the exact references against its reference list.

- **Choudhary et al. (2018)**, *Accident Analysis & Prevention*: speed **variability** on freeways is associated with higher crash risk.
- **Parsa et al. (2020)**, *Accident Analysis & Prevention* (XGBoost + SHAP for real-time crash detection): **speed differences between upstream and downstream locations**, and sudden speed changes, are among the strongest predictors.

Both use loop-detector data at 1–5-minute resolution with detectors a few hundred metres apart. Masar has 10-minute TomTom readings on segments that are kilometres long. The indicator therefore borrows the *factors*, not the models or their thresholds.

---

## 2. Final definition

All parameters are in `SafetyConfig` (`masar_safety.py`).

### 2.1 Data
- Readings are loaded with `masar_features.load_data` at **step 10 min**. On this data the grid keeps every raw reading and only rounds each timestamp to its slot; the actual time stays in `reading_ts`.
- **Collection gaps:** a previous reading counts only if it is at most 1.5 steps older (the same rule as the congestion model's lags).

### 2.2 Components (per point and reading)

| Component | Formula | Parameter |
|---|---|---|
| **Sudden drop** | ratio(t−1) − ratio(t), counted only if ≥ **0.15** (about 15% of free-flow speed). Smaller dips and increases are 0, and it is NaN across a gap. | `min_drop = 0.15` |
| **Speed variability** | CV = std / mean of speed over the window (t − **60 min**, t]. The window **restarts after a collection gap** and needs ≥ 3 readings, otherwise NaN. | `window_min = 60` |
| **Neighbour difference** (approximate) | max over the adjacent points on the same road of \|ratio − neighbour ratio\| at the same time | — |
| **Unusually slow** | max(0, usual ratio for point × hour × day type − ratio) | — |
| *Incident context (optional, off)* | TomTom incidents active within 1 km. Not in the index (§5e). | `use_incidents = False` |

- **Point order** (north → south, from the coordinates): Al Haramain 1 → 2 → 3; Al Madinah 1 → 2 → 3 → 4; King Abdulaziz 3 → 2 → 1. The points are 3–11 km apart.
- **Travel direction is not known**: the CSV has only the segment midpoint. So the neighbour component takes the larger difference to either neighbour.

### 2.3 Index
1. **Scores:** each component → a 0–1 percentile rank **within the point's own history**. **0 stays 0**, and non-zero values are ranked only among that point's non-zero values (`per_point = True`, positive-only ranks).
2. **Combined score:** the weighted mean of the available scores, with weights **variability 1, sudden drop 1, neighbour 0.5, unusually slow 0.5**. It needs ≥ 3 of the 4 components (`min_components = 3`).
3. **Relative to the traffic level:** subtract the mean combined score of all readings at the same **congestion level** (speed_ratio < 0.60 / 0.60–0.85 / ≥ 0.85). Only the excess counts.
4. **Scale:** index = 100 × excess / **0.7**, capped at 100 (`full_scale = 0.7`).
5. **Recovery rule:** at speed_ratio **≥ 0.85** (congestion level Low) the index is 0 unless the reading is itself a sudden drop (`free_flow_ratio = 0.85`). In practice, **High only occurs in slowed traffic**.
6. **Bands** (fixed cut-offs, no quota): **Medium ≥ 21, High ≥ 43**. That is an excess of 0.15 / 0.30, the same meaning the cut-offs 30 / 60 had at the earlier 0.5 scale.
7. **Main reason:** the component with the largest weighted excess over its level mean. In the app it reads "more unstable than usual for this traffic level: *sudden drop*" (or *high variability*, …).

### 2.4 What the web app shows
- **Selected date and hour:** the usual pattern from `outputs/safety_usual.csv`, built from the 10-minute data for each point × day type × hour.
  - **Score:** the **90th percentile** of the index, i.e. "on the worse days at this hour the index reached at least this". It uses the same bands.
  - **Why not the mean:** risk moments are rare and short, so the mean of an hour never exceeds about 20. Every point would always show Low.
  - The card also shows how often that hour was Medium or High.
  - **Result:** 55 of the 480 cells are Medium and 1 is High, mostly at morning onsets (06–07 h) and in the afternoon. Nights are 0.
- **Latest:** the index of each point's newest reading in the training snapshot. It isn't live.

### 2.5 Data after 2 Nov (30-minute collection)
- The indicator and the UI use the **10-minute data** (5–18 Oct).
- **New readings arrive every 30 minutes.** If they are added, the whole history must be put on the same 30-minute grid (`set_step(30)` resamples the 10-minute data to the nearest 30-minute reading), and the indicator recomputed. Its percentiles and level means then come from 30-minute data only. Mixing a 10-minute history with 30-minute new readings would make drops and variability incomparable.
- At 30 minutes the indicator is much weaker (§5g), so this is a known limitation, not a drop-in update.

---

## 3. Final gate (10-minute data)

| Check | Gate | Result |
|---|---|---|
| \|Spearman\| index vs speed_ratio | ≤ 0.65 | **−0.58** ✅ |
| High readings at full free flow (ratio 1.0) | 0 | **0** ✅ |
| High readings right after a collection gap | 0 | **0** ✅ (2 within the following hour, from partial windows) |
| Worst **top-10% overlap**, each weight ±50% | ≥ 80% | **85%** ✅ (variability ×0.5) |
| Night (00–05 h) share of High readings | < 15% | **11%** ✅ |
| *Not gated:* worst fixed-High overlap, weights ±50% | — | 71% (sudden drop ×0.5) |
| *Not gated:* 2-hour window instead of 60 min | — | 52% High / 67% top-10% overlap |

- **High readings:** 109 (0.6%); Medium 623; Low 17,224.
  - **The number of High readings depends on the weights.** Under ±50% on any single weight it ranges from **95 to 167**. The *ranking* of moments is stable (85–99% top-10% overlap); the fixed cut-off moves many readings across 43.
- **Main reason of the High readings:** sudden drop 84, high variability 25.
- **The index is 0 for 87% of readings.**

---

## 4. Confirmed top 10

These are the highest-risk moments, at most one per point within 2 hours. All have "sudden drop" as the main reason.

| # | Point | Time | Index | Speed before → now → after (km/h) | Ratio (level) |
|---|---|---|---|---|---|
| 1 | King Abdulaziz 3 | Wed 14 Oct 20:20 | 89.3 | 83 → 66 → 68 | 0.80 (Medium) |
| 2 | King Abdulaziz 1 | Tue 13 Oct 13:30 | 86.1 | 37 → 15 → 28 | 0.18 (High) |
| 3 | Al Madinah 2 | Sun 11 Oct 18:50 | 82.0 | 94 → 42 → 38 | 0.45 (High) |
| 4 | King Abdulaziz 3 | Wed 7 Oct 23:20 | 78.2 | 83 → 66 → 66 | 0.80 (Medium) |
| 5 | King Abdulaziz 3 | Wed 7 Oct 04:00 | 77.8 | 83 → 65 → 63 | 0.79 (Medium) |
| 6 | King Abdulaziz 1 | Wed 14 Oct 14:20 | 77.5 | 45 → 25 → 37 | 0.30 (High) |
| 7 | Al Madinah 3 | Mon 12 Oct 07:20 | 76.2 | 79 → 52 → 60 | 0.66 (Medium) |
| 8 | King Abdulaziz 2 | Wed 7 Oct 07:20 | 74.7 | 78 → 62 → 62 | 0.79 (Medium) |
| 9 | Al Madinah 2 | Tue 6 Oct 09:50 | 73.7 | 94 → 49 → 94 | 0.52 (High) |
| 10 | Al Haramain 2 | Thu 15 Oct 06:30 | 72.5 | 92 → 67 → 69 | 0.73 (Medium) |

- **What they are:** real slowdowns of 16–52 km/h within one 10-minute step. Most stay slow afterwards; #9 recovers immediately.
- **Spread over the points:** 6 points are represented. King Abdulaziz 3 has 3, all from 83 to about 65 km/h.
- **Night and late evening:** #5 is at 04:00 and #4 at 23:20.
- **Timeline example:** `charts/safety_timeline_example.png` (King Abdulaziz 1, 13 Oct). Only the 13:00 collapse (84 → 15 km/h) is High, with smaller elevations in the 16–19 h stop-and-go. The phase-1 index (dashed) stays high for the whole congested afternoon.

---

## 5. Validation (sanity and robustness only, with no crash data)

### a) Behaviour

| Phase | Readings | Mean index | Share High |
|---|---|---|---|
| Night free flow (00–05 h) | 3,492 | 0.0 | 0% |
| Daytime free flow | 9,177 | 0.0 | 0% |
| **Onset** of a congestion episode (± 1 reading) | 693 | **12.9** | **8.2%** |
| Inside an episode | 2,938 | 5.2 | 0.6% |
| Peak (within 0.05 of the episode minimum) | 835 | 9.2 | 1.1% |
| End of an episode (± 1 reading) | 653 | 4.4 | 0.2% |
| **Short dips** (< 30 min) | 248 | **17.1** | **9.3%** |

**Holds:**
- Zero in free flow by construction.
- Highest at the **onset** of congestion and for short sudden dips, which is where sudden-change risk is expected.
- Steady congestion scores low: it is *usual* for its level.

**Does not hold:** episode **ends** (recovery) score low, because a speed increase isn't counted.

**By hour (weekday mean):** a peak at 07:00 (7.3), and otherwise 2–4 from 12:00 to 23:00. On weekends it is about 0 from 06:00 to 15:00 and 4–5 from 17:00 to 19:00. See `charts/safety_risk_by_hour.png`.

### b) Overlap with congestion
- **Correlation with speed_ratio:** Spearman **−0.58**, against −0.82 for the phase-1 index.
- **Where the High readings fall:** **100% in slowed traffic**, 81% at congestion level Medium and 19% at High, because of the recovery rule.
- **Why the gate was relaxed from 0.3 to 0.65:** TomTom reports speed **capped at free flow**. In free flow every component is 0: no variability is visible, there is no drop, and nothing is below usual. Instability is only measurable once traffic slows, so some link to congestion is unavoidable. It is also consistent with the literature: speed variability and sudden changes occur mainly in dense or transitioning traffic.
  - The aim is not independence from congestion. The aim is to separate **unusual instability** from **ordinary congestion**: steady congestion scores 5.2 on average, onsets 12.9.

### c) Sensitivity (each weight ±50%)

| Variant | Top-10% overlap | Fixed-High overlap | High readings |
|---|---|---|---|
| Variability ×0.5 / ×1.5 | 85% / 93% | 83% / 94% | 103 / 139 |
| Sudden drop ×0.5 / ×1.5 | 98% / 99% | 71% / 82% | 167 / 111 |
| Neighbour ×0.5 / ×1.5 | 94% / 94% | 93% / 83% | 124 / 95 |
| Unusual ×0.5 / ×1.5 | 91% / 92% | 88% / 89% | 106 / 130 |
| Window 120 min (not gated) | 67% | 52% | 68 |

### d) Isolation Forest cross-check
An Isolation Forest (300 trees, 10% contamination) on the per-point scores:
- **All** our High readings fall in the forest's top 10%.
- **61%** of our top 10% are also in its top 10% (34% on the level-adjusted scores).

The forest finds the same extreme moments, and ranks the many small ones differently.

### e) Weak proxy: TomTom incidents (**not accident validation**)
Incidents are mostly jams, i.e. congestion itself, and the data contains **no accident** incidents. Coverage is 10–18 Oct.

| Incidents within 1 km | Readings affected | Mean index near / not near | Share High near / not near |
|---|---|---|---|
| Any | 11,502 (94%) | 1.96 / 0.73 | 0.6% / 0.0% |
| Short-lived jams (≤ 3 h) | 1,634 | 4.14 / 1.55 | 0.9% / 0.5% |
| Closures and road works | 5,732 | 1.94 / 1.85 | 0.7% / 0.4% |

- **Within the same congestion level** the differences are small and mixed. In Medium congestion the index is in fact slightly *lower* near short jams: 6.1 vs 8.1.
- **Why "any incident" can't discriminate:** TomTom keeps some jams "active" for days.
- **Optional incident component:** it would affect 11,502 of the 12,206 covered readings. It stays out of the index.

### f) Top 10
See §4.

### g) 30-minute grid vs 10-minute data

| | 10 min | 30 min |
|---|---|---|
| Readings | 18,036 | 6,056 |
| Share High | 0.6% | 2.0% |
| Spearman vs speed_ratio | −0.58 | −0.55 |
| Largest counted drop | 0.55 | 0.82 |

- Only **38%** of the 10-minute High moments are still High in their 30-minute slot.
- At 30 minutes, a 1-hour window holds just 2 readings, and a drop is measured over 30 minutes. A 10-minute collapse that recovers (like #9) is missed.
- **The window size matters:** with a 2-hour window, only 52% of the High set stays the same. The 60-minute window is therefore part of the definition.

---

## 6. History: what we tried and why we changed it

| Version | Main change | Result | Why it was changed |
|---|---|---|---|
| **Phase 1** | Global percentile ranks, equal weights, no level adjustment (kept as `raw_index`) | Spearman **−0.82**; **88%** of High in congestion; sensitivity 91–99%; top 10 all from the two shortest segments (King Abdulaziz 1, Al Madinah 2) | It mostly restated the congestion level. It was also biased to short, jumpy segments. |
| **9-band attempt** | Per-point percentiles; excess over the mean at the same speed band (9 bands); weights 1 / 1 / 0.5 / 0.5 | Spearman −0.17, but **76% of High at full free flow**. Worst High overlap **42%**; 2 h window 67%. Top 10 all recovery moments or the first reading after a gap. | Free flow's expected score was near 0, so leftover variability after a slowdown became "High". |
| **Option 1** | 3 congestion levels; recovery rule (0 at ratio ≥ 0.95 unless a drop); variability window restarts after a gap | Gate passed (−0.61, 0 at full free flow, 0 after a gap, worst overlap 84%) | **Top 10 artefact:** with 84% of drops at zero, a 1 km/h dip (71 → 70) scored 100, and 380 readings were tied at 100. |
| **Meaningful-change scoring** | Positive-only percentiles; drops count only if ≥ 0.15; scale 0.7 | Top 10 became real drops (confirmed), with 0 ties at 100. But with High ≥ 60 only 44 readings were High, and the worst overlap was 39% (2 h window). | The wider scale changed what "60" meant. High was moved to 43 (same excess as before), and the window was fixed at 60 min as part of the definition. |
| **High ≥ 43, weights-only gate** | — | 246 High; worst fixed-High overlap **62%**; **25% of High at night** (51 of 61 "high variability") | Night High came from small oscillations (median −1 km/h) on King Abdulaziz 2 and 3, at 0.85–0.88 of free flow. |
| **Variability minimum** (std of ratio ≥ 0.05–0.08) | Variability counts only above a minimum | Night share 15 → 8%, but worst overlap 54–58%, High in congestion 86–98%, and 6–8 of the confirmed top 10 kept | Night oscillations have nearly the same ratio std as real onsets (median 0.062 vs 0.073). A minimum can't separate them. Left in the code, off (0.0). |
| **Option c** (drop-driven weights) | Variability 0.25 or info-only | Worst overlap 70% / 67% (drop ×0.5); night 17% / 14%; info-only lets 12 post-gap readings through | When one sparse component carries the index, the fixed High set depends on that component's weight. |
| **Final** | Recovery rule from **0.85**; sensitivity gated on the **ranking** (top 10%) | All gate checks pass (§3); the confirmed top 10 is unchanged | — |

The four level-adjustment variants on the final scores (`variant_comparison` in the validation JSON) are compared below. All other settings are final; the chosen variant uses recovery from 0.85, and the recovery variants there use the 0.85 rule.

| Variant | Spearman | High at full free flow | High in congestion | Worst High overlap | Worst top-10% overlap |
|---|---|---|---|---|---|
| 9 speed bands | −0.18 | 75% | 25% | 35% | 75% |
| 3 congestion levels | −0.23 | 54% | 20% | 36% | 81% |
| 9 bands + recovery rule | −0.59 | 0% | 100% | 68% | 84% |
| **3 levels + recovery rule (final)** | **−0.58** | **0%** | **100%** | **71%** | **85%** |

**Before / after:**

| | Phase 1 (`raw_index`) | Final (`risk_index`) |
|---|---|---|
| Spearman vs speed_ratio | −0.82 | −0.58 |
| High readings in congestion | 88% | 100% (by design: High only in slowed traffic) |
| High readings at full free flow | ~1% | 0 |
| Sensitivity (weights ±50%) | ≥ 91% (top 10%) | ≥ 85% (top 10%) |
| Steady congestion vs onset (mean index) | 62 vs 63 | 5.2 vs 12.9 |
| Top-10 points | King Abdulaziz 1 ×6, Al Madinah 2 ×4 | 6 different points |

High share per point (%):

| Point | Phase 1 (top 10%) | Final (High band) |
|---|---|---|
| Al Haramain 1 | 8.1 | 0.2 |
| Al Haramain 2 | 17.4 | 0.8 |
| Al Haramain 3 | 10.6 | 0.7 |
| Al Madinah 1 | 3.1 | 0.1 |
| Al Madinah 2 | 2.3 | 0.3 |
| Al Madinah 3 | 14.2 | 0.6 |
| Al Madinah 4 | 9.1 | 0.7 |
| King Abdulaziz 1 | 10.5 | 1.0 |
| King Abdulaziz 2 | 13.9 | 1.1 |
| King Abdulaziz 3 | 10.4 | 0.6 |

---

## 7. Limitations
- **No crash validation.** There is no road-level crash data. The indicator shows *speed patterns linked to risk in the literature*, not measured risk, and must not be presented as accident prediction.
- **Time step.** 10 min now and 30 min after the collector restart, against 1–5 min in the literature. Shock waves shorter than a step are invisible. At 30 min only 38% of the 10-minute High moments survive (§5g). The 60-minute window is a defined parameter: with 2 hours the High set changes by about half.
- **Km-spaced points and long segments.** Points are 3–11 km apart and TomTom segments are 2–20 km long, sometimes overlapping. The neighbour component is approximate, and the travel direction is unknown.
- **Speed capped at free flow.** TomTom reports at most the free-flow speed, so **speeding and high-speed instability are invisible**. Together with the recovery rule, this means **High only occurs in slowed traffic**.
- **Chosen weights and cut-offs.** The weights (1 / 1 / 0.5 / 0.5), the drop minimum (0.15), the scale (0.7) and the bands (21 / 43) are judgment values, not learned from crashes. **The number of High readings depends on the weights** (95–167 under ±50%), even though the ranking is stable.
- **Recovery is not scored.** The end of a congestion episode scores low.
- **Night oscillations.** Some points (King Abdulaziz 2 and 3) oscillate below their free-flow speed at night. The recovery rule removes most of this, but 11% of High readings are still at night.
- **In-sample.** The usual tables, percentiles and level means are fitted on the same 14 days they describe.

## 8. Future work
- **Validate, and learn the weights, once crash data is available** (e.g. police or Najm records with time and location), instead of the judgment weights.
- **A supervised model following Parsa et al.:** XGBoost with SHAP explanations, using these components as features and crashes as the target.
- **Finer time steps** (≤ 5 min) once the API budget allows, so short shock waves become visible.
- **Closer points:** pairs a few hundred metres apart around known bottlenecks, with known travel direction, for a real upstream/downstream difference.
- **Score recovery too**, e.g. sharp speed increases as a smaller component.
- **Risk forecasting** from the existing +30 / +60 min speed forecasts.
