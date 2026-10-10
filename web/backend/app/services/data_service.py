from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class DataService:
    def __init__(self) -> None:
        self.catalog = pd.read_csv(PROCESSED_DIR / "road_catalog.csv")
        self.catalog["road_key"] = self.catalog["road_key"].astype(str)
        self.catalog_index = self.catalog.set_index("road_key", drop=False)

        self.accident_hour = pd.read_csv(
            PROCESSED_DIR / "accident_road_hour_summary.csv.gz"
        )
        self.accident_hour["road_key"] = self.accident_hour["road_key"].astype(str)

        self.traffic_patterns = pd.read_csv(
            PROCESSED_DIR / "traffic_patterns.csv.gz"
        )
        self.traffic_patterns["road_key"] = self.traffic_patterns["road_key"].astype(str)

        self.road_hour_stats = pd.read_csv(
            PROCESSED_DIR / "road_hour_stats.csv.gz"
        )
        self.road_hour_stats["road_key"] = self.road_hour_stats["road_key"].astype(str)

        segment_path = PROCESSED_DIR / "segment_catalog.csv"
        self.segment_catalog = pd.read_csv(segment_path, dtype={"segment_id": str}) if segment_path.exists() else pd.DataFrame()
        if not self.segment_catalog.empty:
            self.segment_catalog["road_key"] = self.segment_catalog["road_key"].astype(str)
            self.segment_catalog["segment_id"] = self.segment_catalog["segment_id"].astype(str)

        causes_path = PROCESSED_DIR / "accident_causes.csv.gz"
        self.accident_causes = pd.read_csv(causes_path) if causes_path.exists() else pd.DataFrame()
        if not self.accident_causes.empty:
            self.accident_causes["road_key"] = self.accident_causes["road_key"].astype(str)

        time_path = PROCESSED_DIR / "accident_time_profiles.csv"
        self.time_profiles = pd.read_csv(time_path) if time_path.exists() else pd.DataFrame()
        if not self.time_profiles.empty:
            self.time_profiles["road_key"] = self.time_profiles["road_key"].astype(str)
            self.time_profile_index = self.time_profiles.set_index("road_key", drop=False)
        else:
            self.time_profile_index = pd.DataFrame()

        priority_path = PROCESSED_DIR / "government_priority.csv"
        self.government_priority = pd.read_csv(priority_path) if priority_path.exists() else pd.DataFrame()
        if not self.government_priority.empty:
            self.government_priority["road_key"] = self.government_priority["road_key"].astype(str)

        self.model_bundle = None
        self.model_load_error: str | None = None
        try:
            self.model_bundle = joblib.load(MODEL_DIR / "traffic_model.joblib")
        except Exception as exc:  # robust across local sklearn versions
            self.model_load_error = f"{type(exc).__name__}: {exc}"

        self.accident_lookup = {
            (str(r.road_key), int(r.day_of_week), int(r.hour)): {
                "historical_crashes": int(r.historical_crashes),
                "persons_injured": int(r.persons_injured),
                "persons_killed": int(r.persons_killed),
            }
            for r in self.accident_hour.itertuples(index=False)
        }

        self.max_crashes_by_road = (
            self.accident_hour.groupby("road_key")["historical_crashes"].max().to_dict()
        )

        self.pattern_lookup = {
            (str(r.road_key), int(r.month), int(r.day_of_week), int(r.hour)): {
                "median": float(r.median_traffic),
                "mean": float(r.mean_traffic),
                "p25": float(r.p25),
                "p75": float(r.p75),
                "observations": int(r.observations),
            }
            for r in self.traffic_patterns.itertuples(index=False)
        }

        self.road_hour_lookup = {
            (str(r.road_key), int(r.day_of_week), int(r.hour)): {
                "median": float(r.median_traffic),
                "mean": float(r.mean_traffic),
                "observations": int(r.observations),
            }
            for r in self.road_hour_stats.itertuples(index=False)
        }

    @property
    def model_available(self) -> bool:
        return self.model_bundle is not None

    def road(self, road_key: str) -> dict | None:
        if road_key not in self.catalog_index.index:
            return None
        row = self.catalog_index.loc[road_key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        data = row.to_dict()
        for key, value in list(data.items()):
            if pd.isna(value):
                data[key] = None
        data["safety_score"] = round(max(0.0, min(100.0, 100 - float(data.get("base_risk_score") or 0))), 1)
        return data

    def search_roads(self, query: str = "", limit: int = 25) -> list[dict]:
        df = self.catalog
        if query:
            q = query.strip().upper()
            df = df[
                df["display_name"].fillna("").str.upper().str.contains(q, regex=False)
                | df["road_key"].str.upper().str.contains(q, regex=False)
            ]
        df = df.sort_values(
            ["historical_crashes", "observations"], ascending=[False, False]
        ).head(limit)
        return [
            {
                "road_key": str(r.road_key),
                "display_name": str(r.display_name),
                "historical_crashes": int(r.historical_crashes),
                "risk_level": str(r.risk_level),
                "risk_score": round(float(r.base_risk_score), 1),
                "safety_score": round(100 - float(r.base_risk_score), 1),
                "avg_traffic": round(float(r.avg_traffic), 1),
                "latitude": None if pd.isna(r.median_latitude) else float(r.median_latitude),
                "longitude": None if pd.isna(r.median_longitude) else float(r.median_longitude),
            }
            for r in df.itertuples(index=False)
        ]

    def map_roads(self) -> list[dict]:
        df = self.catalog.dropna(subset=["median_latitude", "median_longitude"]).copy()
        return [
            {
                "road_key": str(r.road_key),
                "display_name": str(r.display_name),
                "latitude": float(r.median_latitude),
                "longitude": float(r.median_longitude),
                "historical_crashes": int(r.historical_crashes),
                "risk_score": round(float(r.base_risk_score), 1),
                "safety_score": round(100 - float(r.base_risk_score), 1),
                "risk_level": str(r.risk_level),
                "avg_traffic": round(float(r.avg_traffic), 1),
            }
            for r in df.itertuples(index=False)
        ]

    def segments(self) -> list[dict]:
        if self.segment_catalog.empty:
            return []
        df = self.segment_catalog.copy()
        records = []
        for r in df.itertuples(index=False):
            records.append({
                "segment_id": str(r.segment_id),
                "road_key": str(r.road_key),
                "display_name": str(r.display_name) if not pd.isna(r.display_name) else str(r.roadway_name),
                "roadway_name": str(r.roadway_name),
                "from_street": None if pd.isna(r.from_street) else str(r.from_street),
                "to_street": None if pd.isna(r.to_street) else str(r.to_street),
                "direction": None if pd.isna(r.direction) else str(r.direction),
                "avg_traffic": 0.0 if pd.isna(r.avg_traffic) else round(float(r.avg_traffic), 1),
                "historical_crashes": 0 if pd.isna(r.historical_crashes) else int(r.historical_crashes),
                "observations": int(r.observations),
                "risk_score": 0.0 if pd.isna(r.base_risk_score) else round(float(r.base_risk_score), 1),
                "safety_score": 100.0 if pd.isna(r.safety_score) else round(float(r.safety_score), 1),
                "risk_level": "Low" if pd.isna(r.risk_level) else str(r.risk_level),
                "latitude": None if pd.isna(r.median_latitude) else float(r.median_latitude),
                "longitude": None if pd.isna(r.median_longitude) else float(r.median_longitude),
            })
        return records

    def road_insights(self, road_key: str) -> dict | None:
        road = self.road(road_key)
        if road is None:
            return None

        risk_score = float(road.get("base_risk_score") or 0)
        safety_score = round(100 - risk_score, 1)
        if safety_score >= 70:
            safety_level = "Safer"
        elif safety_score >= 45:
            safety_level = "Moderate"
        else:
            safety_level = "Needs attention"

        # Hour/day distributions
        road_hours = self.accident_hour[self.accident_hour["road_key"] == road_key]
        hourly = (
            road_hours.groupby("hour", as_index=False)["historical_crashes"].sum()
            .set_index("hour")
            .reindex(range(24), fill_value=0)
        )
        hourly_distribution = [
            {"hour": int(h), "crashes": int(v)}
            for h, v in hourly["historical_crashes"].items()
        ]

        daily = (
            road_hours.groupby("day_of_week", as_index=False)["historical_crashes"].sum()
            .set_index("day_of_week")
            .reindex(range(7), fill_value=0)
        )
        day_distribution = [
            {"day": DAY_NAMES[int(d)], "day_of_week": int(d), "crashes": int(v)}
            for d, v in daily["historical_crashes"].items()
        ]

        time_profile = None
        if not self.time_profiles.empty and road_key in self.time_profile_index.index:
            tr = self.time_profile_index.loc[road_key]
            if isinstance(tr, pd.DataFrame):
                tr = tr.iloc[0]
            time_profile = {
                "peak_window_start": int(tr.peak_window_start),
                "peak_window_end": int(tr.peak_window_end),
                "peak_window_crashes": int(tr.peak_window_crashes),
                "top_day_name": str(tr.top_day_name),
                "weekend_share_pct": float(tr.weekend_share_pct),
            }

        # Causes: prefer specified factors; keep Unspecified only as fallback.
        causes: list[dict] = []
        if not self.accident_causes.empty:
            cdf = self.accident_causes[self.accident_causes["road_key"] == road_key].copy()
            specified = cdf[cdf["cause"].str.upper() != "UNSPECIFIED"].sort_values("crashes", ascending=False)
            chosen = specified.head(5) if not specified.empty else cdf.sort_values("crashes", ascending=False).head(5)
            causes = [
                {
                    "cause": str(r.cause),
                    "crashes": int(r.crashes),
                    "share_pct": float(r.share_pct),
                }
                for r in chosen.itertuples(index=False)
            ]

        return {
            "road_key": road_key,
            "display_name": road.get("display_name") or road_key,
            "safety_score": safety_score,
            "safety_level": safety_level,
            "risk_score": round(risk_score, 1),
            "risk_level": road.get("risk_level") or "Low",
            "historical_crashes": int(road.get("historical_crashes") or 0),
            "persons_injured": int(road.get("persons_injured") or 0),
            "persons_killed": int(road.get("persons_killed") or 0),
            "peak_hour": None if road.get("peak_hour") is None else int(road["peak_hour"]),
            "accident_time": time_profile,
            "hourly_accidents": hourly_distribution,
            "day_accidents": day_distribution,
            "top_causes": causes,
            "data_coverage": {
                "traffic_observations": int(road.get("observations") or 0),
                "first_traffic_observation": road.get("first_observation"),
                "last_traffic_observation": road.get("last_observation"),
                "first_crash": road.get("first_crash"),
                "last_crash": road.get("last_crash"),
            },
            "score_note": "Road Safety Score is a historical decision-support indicator. Higher is safer; it is not a guarantee of future safety.",
        }

    def priority_roads(self, limit: int = 8) -> list[dict]:
        if self.government_priority.empty:
            df = self.catalog.copy()
            df["government_priority_score"] = df["base_risk_score"]
            df["government_priority_level"] = df["risk_level"]
        else:
            df = self.government_priority.copy()
        df = df.sort_values(["government_priority_score", "historical_crashes"], ascending=False).head(limit)
        out = []
        for r in df.itertuples(index=False):
            out.append({
                "road_key": str(r.road_key),
                "display_name": str(r.display_name),
                "priority_score": round(float(r.government_priority_score), 1),
                "priority_level": str(r.government_priority_level),
                "avg_traffic": round(float(r.avg_traffic), 1),
                "historical_crashes": int(r.historical_crashes),
                "persons_injured": int(r.persons_injured),
                "persons_killed": int(r.persons_killed),
                "risk_score": round(float(r.base_risk_score), 1),
                "risk_level": str(r.risk_level),
                "latitude": None if pd.isna(r.median_latitude) else float(r.median_latitude),
                "longitude": None if pd.isna(r.median_longitude) else float(r.median_longitude),
            })
        return out


@lru_cache(maxsize=1)
def get_data_service() -> DataService:
    return DataService()
