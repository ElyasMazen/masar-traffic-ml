"""
Serves the Masar SAFETY RISK INDICATOR (not accident prediction) for the Jeddah view.

Reads what safety_risk.py wrote at the repo root:
  outputs/safety_usual.csv       usual risk per point x day type x hour (10-min data)
  outputs/safety_risk.csv        per reading; used for each point's latest risk
  outputs/safety_validation.json definition and gate numbers

For a chosen date and hour the score is the 90th percentile of the index at
that point, day type and hour ("on the worse days at this hour"); the hourly
mean of these rare, short events never reaches Medium. See
docs/SAFETY_INDICATOR.md. Restart the backend after re-running safety_risk.py.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = REPO_ROOT / "outputs"
WEEKEND_DAYS = (4, 5)   # Friday, Saturday (Python weekday), as in masar_features

NOTE = ("Indicator based on speed patterns linked to crash risk in the literature; "
        "not validated against crash data.")
REASON_PREFIX = "More unstable than usual for this traffic level"


def _component(value) -> str:
    """main_reason from the CSV; empty cells come back as NaN."""
    return value if isinstance(value, str) and value else "none"


def _reason_text(component: str) -> str:
    if component == "none":
        return "Not more unstable than usual for this traffic level"
    return f"{REASON_PREFIX}: {component}"


class SafetyService:
    def __init__(self) -> None:
        self.usual = pd.read_csv(OUTPUTS / "safety_usual.csv")
        self.validation = json.loads((OUTPUTS / "safety_validation.json").read_text(encoding="utf-8"))
        cfg = self.validation["config"]
        self.cutoffs = {"medium": cfg["band_cutoffs_medium_high"][0], "high": cfg["band_cutoffs_medium_high"][1]}

        readings = pd.read_csv(OUTPUTS / "safety_risk.csv",
                               usecols=["point_id", "ts", "reading_ts", "risk_index", "band", "main_reason",
                                        "speed_kmh", "speed_ratio"])
        readings = readings.dropna(subset=["risk_index"])
        latest = readings.sort_values("ts").groupby("point_id").tail(1)
        self.latest_rows = {
            r.point_id: {
                "point_id": r.point_id,
                "reading_time": str(r.reading_ts).replace(" ", "T"),
                "score": round(float(r.risk_index), 1),
                "band": r.band,
                "main_component": _component(r.main_reason),
                "reason": _reason_text(_component(r.main_reason)),
                "speed_kmh": float(r.speed_kmh),
            }
            for r in latest.itertuples(index=False)
        }

    def levels(self, sim_date: date, hour: int) -> list[dict]:
        """Usual safety risk of every point for that date's day type and hour."""
        daytype = "weekend" if sim_date.weekday() in WEEKEND_DAYS else "weekday"
        rows = self.usual[(self.usual["daytype"] == daytype) & (self.usual["hour"] == hour)]
        out = []
        for r in rows.itertuples(index=False):
            # A score of 0 means nothing beyond usual at this hour, whatever the rare elevated readings were.
            component = _component(r.main_reason) if r.p90_index > 0 else "none"
            out.append({
            "point_id": r.point_id,
            "day_type": daytype,
            "hour": int(hour),
            "score": round(float(r.p90_index), 1),
            "band": r.band,
            "main_component": component,
            "reason": _reason_text(component),
            "share_elevated": round(float(r.share_medium_or_high), 3),
            "share_high": round(float(r.share_high), 3),
            "readings": int(r.readings),
            "days": int(r.days),
            })
        return out

    def latest(self) -> list[dict]:
        return list(self.latest_rows.values())

    def summary(self) -> dict:
        v = self.validation
        return {
            "name": "Safety risk",
            "note": NOTE,
            "definition": "Risk beyond what is usual for this traffic level, from sudden drops, speed variability, "
                          "the speed difference to the neighbouring point and unusual slowness "
                          "(see docs/SAFETY_INDICATOR.md).",
            "score_for_date_hour": "90th percentile of the index at this point, day type and hour "
                                   "(on the worse days at this hour)",
            "cutoffs": self.cutoffs,
            "data": v["data"],
            "gate": v["gate"],
        }


@lru_cache(maxsize=1)
def get_safety_service() -> SafetyService:
    return SafetyService()
