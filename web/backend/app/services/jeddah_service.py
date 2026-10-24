"""
Serves the Jeddah view straight from the masar-traffic-ml repo this app lives in.

It imports the repo's own masar_features.py and reads models/ and
outputs/results.json from the repo root, so serving uses exactly what
train_masar.py produced. Retrain with scripts/build_jeddah.py --retrain.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# web/backend/app/services/jeddah_service.py -> repo root is four levels up.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import masar_features as mf  # noqa: E402  (the shared training module at the repo root)

RESULTS_PATH = REPO_ROOT / "outputs" / "results.json"
MODEL_DIR = REPO_ROOT / "models"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _road_short(road: str) -> str:
    return road.replace(" Road", " Rd")


class JeddahService:
    def __init__(self) -> None:
        self.results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        # Use the grid step the models were trained with (older results: 10 min).
        self.step = int(self.results.get("step", {}).get("step_min", 10))
        mf.set_step(self.step)
        # The exact snapshot the models were trained on (repo-relative path in results.json).
        self.readings = mf.prepare(pd.read_csv(REPO_ROOT / self.results["data"]["snapshot"]))
        self.weekdays_with_data = set(self.results["data"]["weekdays_with_data"])
        self.catalog = self._build_catalog()
        self.time_state = joblib.load(MODEL_DIR / "time_of_day.joblib")

        self.horizon = int(self.results["short_term"]["selected"]["horizon_min"])
        self.short_model = joblib.load(MODEL_DIR / f"short_term_rf_{self.horizon}min.joblib")
        self.short_term = self._short_term_forecast()

    # ------------------------------------------------------------------ points

    def _build_catalog(self) -> dict[str, dict]:
        catalog = {}
        for pid, g in self.readings.groupby("point_id"):
            weekday = g[g["daytype"] == "weekday"]
            hourly = (weekday if not weekday.empty else g).groupby("hour")["speed_ratio"].mean()
            road = str(g["road"].iloc[0])
            index = int(str(pid).rsplit("_", 1)[1])
            catalog[str(pid)] = {
                "point_id": str(pid),
                "road": road,
                "point_index": index,
                "label": f"{road} · Point {index}",
                "short_label": f"{_road_short(road)} · Point {index}",
                "latitude": float(g["lat"].iloc[0]),
                "longitude": float(g["lon"].iloc[0]),
                "frc": str(g["frc"].mode().iloc[0]),
                "free_flow_speed_kmh": float(g["free_flow_speed_kmh"].median()),
                "free_flow_travel_time_s": float(g["free_flow_travel_time_s"].median()),
                "weekday_avg_speed_kmh": round(float((weekday if not weekday.empty else g)["current_speed_kmh"].mean()), 1),
                "weekday_peak_hour": int(hourly.idxmin()),
                "weekday_peak_level": mf.classify(float(hourly.min())),
            }
        return catalog

    def point(self, point_id: str) -> dict | None:
        return self.catalog.get(point_id)

    def points(self) -> list[dict]:
        return list(self.catalog.values())

    # ------------------------------------------------------- time of day

    def _ratios(self, rows: list[tuple[str, datetime]]) -> np.ndarray:
        """Hour-level speed ratio: mean over the grid slots in each hour (2 at step 30, 6 at step 10)."""
        minutes = range(0, 60, self.step)
        frame = pd.DataFrame([
            {
                "point_id": pid,
                "ts": dt.replace(minute=minute),
                "free_flow_speed_kmh": self.catalog[pid]["free_flow_speed_kmh"],
                "frc": self.catalog[pid]["frc"],
            }
            for pid, dt in rows
            for minute in minutes
        ])
        return mf.time_of_day_ratio(self.time_state, frame).reshape(len(rows), len(minutes)).mean(axis=1)

    def levels(self, sim_date: date, hour: int) -> list[dict]:
        dt = datetime.combine(sim_date, datetime.min.time()).replace(hour=hour)
        ids = list(self.catalog)
        out = []
        for pid, ratio in zip(ids, self._ratios([(pid, dt) for pid in ids])):
            out.append({
                "point_id": pid,
                # Unrounded: the page truncates it for display so it always agrees with the level.
                "speed_ratio": float(ratio),
                "speed_kmh": round(float(ratio) * self.catalog[pid]["free_flow_speed_kmh"], 1),
                "traffic_level": mf.classify(float(ratio)),
            })
        return out

    def simulate(self, point_id: str, sim_date: date, hour: int) -> dict:
        point = self.point(point_id)
        if point is None:
            raise KeyError(point_id)

        dt = datetime.combine(sim_date, datetime.min.time()).replace(hour=hour)
        hours = [dt + timedelta(hours=k) for k in range(12)]
        ratios = self._ratios([(point_id, h) for h in hours])
        ratio = float(ratios[0])
        ffs = point["free_flow_speed_kmh"]
        fftt = point["free_flow_travel_time_s"]
        travel = fftt / max(ratio, 0.05)

        daytype = "weekend" if dt.weekday() in mf.WEEKEND_DAYS else "weekday"
        usual = self.time_state["usual"]
        key = (point_id, hour, daytype)
        p25 = usual["point_hour_daytype_p25"].get(key)
        p75 = usual["point_hour_daytype_p75"].get(key)
        days = int(usual["point_hour_daytype_days"].get(key, 0))

        return {
            "point": point,
            "datetime": dt.isoformat(),
            "day_type": daytype,
            "predicted_speed_kmh": round(ratio * ffs, 1),
            "predicted_speed_ratio": ratio,
            "free_flow_speed_kmh": round(ffs, 1),
            "traffic_level": mf.classify(ratio),
            # Exact seconds; the page rounds the delay once, at the end.
            "expected_travel_time_s": travel,
            "free_flow_travel_time_s": fftt,
            "historical_range": {
                "lower_speed_kmh": None if p25 is None else round(p25 * ffs, 1),
                "upper_speed_kmh": None if p75 is None else round(p75 * ffs, 1),
                "days_observed": days,
            },
            "data_coverage": self._coverage(days, dt.weekday(), daytype),
            "method": {"key": self.time_state["method"], "label": self.time_state["label"]},
            "forecast": [
                {
                    "datetime": h.isoformat(),
                    "hour": h.hour,
                    "speed_kmh": round(float(r) * ffs, 1),
                    "speed_ratio": float(r),
                    "level": mf.classify(float(r)),
                }
                for h, r in zip(hours, ratios)
            ],
            "short_term": next((p for p in self.short_term["points"] if p["point_id"] == point_id), None),
            "short_term_horizon_min": self.horizon,
        }

    def _coverage(self, days: int, dow: int, daytype: str) -> dict:
        """How many days of readings back this point/hour/day type - a count, not a confidence."""
        kind = "weekend day" if daytype == "weekend" else "weekday"
        summary = f"{days} {kind}{'s' if days != 1 else ''} observed at this hour"
        notes = []
        if dow not in self.weekdays_with_data:
            if daytype == "weekday":
                notes.append(
                    f"No {DAY_NAMES[dow]} has been collected yet, so this uses the weekday (Sun-Thu) "
                    "pattern for this hour - never weekend days."
                )
            else:
                notes.append(f"No {DAY_NAMES[dow]} has been collected yet, so this uses the Fri/Sat weekend pattern.")
        if days == 0:
            notes.append(
                f"There are no {daytype} readings for this point at this hour, so it falls back to the "
                "all-days average for this hour."
            )
        return {"days_observed": days, "summary": summary, "note": " ".join(notes)}

    # ---------------------------------------------------------- short term

    def _short_term_forecast(self) -> dict:
        bundle = self.short_model
        x = mf.add_usual(mf.latest_samples(self.readings, self.horizon), bundle["usual"])
        model = bundle["pipeline"]
        proba = model.predict_proba(x[bundle["features"]])
        classes = list(model.classes_)

        points = []
        for i, row in enumerate(x.itertuples(index=False)):
            points.append({
                "point_id": row.point_id,
                "latest_timestamp": pd.Timestamp(row.ts).isoformat(),            # grid slot
                "latest_reading_time": pd.Timestamp(row.reading_ts).isoformat(),  # actual collection time
                "forecast_for": pd.Timestamp(row.target_time).isoformat(),
                "latest_speed_ratio": float(row.speed_ratio_t),
                "persistence_level": row.congestion_level,
                "predicted_level": classes[int(proba[i].argmax())],
                "model_probability": round(float(proba[i].max()), 3),
                "probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba[i])},
            })
        return {
            "horizon_min": self.horizon,
            "step_min": self.step,
            "model": "Random Forest (class_weight=balanced)",
            "latest_timestamp": max(p["latest_timestamp"] for p in points),
            "selected": self.results["short_term"]["selected"],
            "points": points,
        }

    # ------------------------------------------------------------- summary

    def summary(self) -> dict:
        r = self.results
        audit = r["data"]["audit"]

        return {
            "data": {
                **{k: r["data"][k] for k in (
                    "source", "rows", "rows_raw", "points", "first_reading", "last_reading", "days",
                    "day_names", "trained_through", "loaded_at",
                )},
                "weekdays": audit["weekdays"],
                "weekend_days": audit["weekend_days"],
                "step_min": self.step,
                "horizons_min": r.get("step", {}).get("horizons_min", [10, 30, 60]),
            },
            "models": r["models"],
            "splits": r["splits"],
            # Rows for subset "all", plus "weekday" / "weekend" when a test set holds both.
            "short_term": {
                "methods": r["short_term"]["methods"],
                "selected": r["short_term"]["selected"],
                "results": r["short_term"]["results"],
            },
            "time_of_day": {
                "methods": r["time_of_day"]["methods"],
                "selected": r["time_of_day"]["selected"],
                "results": r["time_of_day"]["results"],
            },
        }


@lru_cache(maxsize=1)
def get_jeddah_service() -> JeddahService:
    return JeddahService()
