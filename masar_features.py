"""
Shared Masar data loading, feature engineering and baselines.

Used by train_masar.py, predict_next.py and the web app's backend
(web/backend imports this file from the repo root), so training and serving
cannot drift apart.

Reading the raw GitHub CSV is read-only. Never call the TomTom API from here
and never write to the data repo.
"""

from __future__ import annotations

import io
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/elAmigoOs/tomtomdata/main/jeddah_tomtom_flow.csv"

LABELS = ["Low", "Medium", "High"]
LOW_MIN_RATIO = 0.85      # speed_ratio >= 0.85 -> Low
MEDIUM_MIN_RATIO = 0.60   # 0.60-0.85 -> Medium ; < 0.60 -> High
WEEKEND_DAYS = (4, 5)     # Friday, Saturday (pandas dayofweek)

STEP_MIN = 10             # collection interval
TOLERANCE_MIN = 5         # a reading counts as "t + h" if within +/- 5 min
N_LAGS = 6
HORIZONS_MIN = (10, 30, 60)


# --------------------------------------------------------------------- labels

def classify(speed_ratio: float) -> str:
    if speed_ratio >= LOW_MIN_RATIO:
        return "Low"
    if speed_ratio >= MEDIUM_MIN_RATIO:
        return "Medium"
    return "High"


def classify_many(ratios) -> np.ndarray:
    r = np.asarray(ratios, dtype=float)
    return np.where(r >= LOW_MIN_RATIO, "Low", np.where(r >= MEDIUM_MIN_RATIO, "Medium", "High"))


def day_type(dow) -> np.ndarray:
    return np.where(np.isin(np.asarray(dow), WEEKEND_DAYS), "weekend", "weekday")


# ----------------------------------------------------------------------- data

def load_data(csv: str | Path | None = None, snapshot_dir: str | Path | None = None):
    """Load TomTom readings. The one place data enters the pipeline.

    csv=None  -> download the raw GitHub CSV (can lag ~5 min behind the latest
                 commit) and, if snapshot_dir is given, save the exact bytes as
                 a dated snapshot so the run can be reproduced.
    csv=path  -> read a local snapshot instead.

    Returns (readings, info).
    """
    if csv is None:
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            raw = response.read()
        source, snapshot = DATA_URL, None
        if snapshot_dir is not None:
            folder = Path(snapshot_dir)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"jeddah_tomtom_flow_{datetime.now():%Y%m%d_%H%M%S}.csv"
            path.write_bytes(raw)
            snapshot = str(path.resolve())
        frame = pd.read_csv(io.BytesIO(raw))
    else:
        source = snapshot = str(Path(csv).resolve())
        frame = pd.read_csv(csv)

    d = prepare(frame)
    info = {
        "source": source,
        "snapshot": snapshot,
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(d)),
        "points": int(d["point_id"].nunique()),
        "first_reading": str(d["ts"].min()),
        "last_reading": str(d["ts"].max()),
        "days": int(d["ts"].dt.normalize().nunique()),
    }
    return d, info


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    d["ts"] = pd.to_datetime(d["timestamp_local"])
    d = d.drop_duplicates(["point_id", "ts"]).sort_values(["point_id", "ts"]).reset_index(drop=True)
    d["hour"] = d["ts"].dt.hour
    d["dow"] = d["ts"].dt.dayofweek
    d["daytype"] = day_type(d["dow"])
    d["congestion_level"] = classify_many(d["speed_ratio"])
    return d


# ---------------------------------------------------- historical average

def fit_usual(readings: pd.DataFrame) -> dict:
    """"Usual" speed ratio tables. Fit on training readings only."""
    date = readings["ts"].dt.date
    by_dt = readings.groupby(["point_id", "hour", "daytype"])
    return {
        "global": float(readings["speed_ratio"].mean()),
        "point": readings.groupby("point_id")["speed_ratio"].mean().to_dict(),
        "point_hour": readings.groupby(["point_id", "hour"])["speed_ratio"].mean().to_dict(),
        "point_hour_daytype": by_dt["speed_ratio"].mean().to_dict(),
        "point_hour_daytype_days": readings.assign(date=date)
        .groupby(["point_id", "hour", "daytype"])["date"].nunique().to_dict(),
        "point_hour_daytype_p25": by_dt["speed_ratio"].quantile(0.25).to_dict(),
        "point_hour_daytype_p75": by_dt["speed_ratio"].quantile(0.75).to_dict(),
    }


def usual_ratio(usual: dict, point_ids, hours, daytypes, by_daytype: bool = True) -> np.ndarray:
    """Point x hour (x weekday/weekend) average with fallbacks.

    A weekday with no data of its own (e.g. Sunday) still uses the WEEKDAY
    average for that hour, never weekend days. Only if a point has no reading
    at all for that hour and day type does it fall back to the all-days
    point x hour average, then the point average.
    """
    out = np.empty(len(point_ids), dtype=float)
    for i, (p, h, t) in enumerate(zip(point_ids, hours, daytypes)):
        h = int(h)
        value = usual["point_hour_daytype"].get((p, h, t)) if by_daytype else None
        if value is None:
            value = usual["point_hour"].get((p, h))
        if value is None:
            value = usual["point"].get(p, usual["global"])
        out[i] = value
    return out


# --------------------------------------------------- time-of-day HGB model

TIME_MODEL_FEATURES = [
    "minute_sin", "minute_cos", "is_weekend",
    "free_flow_speed_kmh", "frc_code", "point_mean", "point_hour_mean",
]


def frc_code(frc) -> int:
    digits = "".join(ch for ch in str(frc) if ch.isdigit())
    return int(digits) if digits else 9


def time_model_features(frame: pd.DataFrame, usual: dict) -> pd.DataFrame:
    """frame needs point_id, ts, free_flow_speed_kmh, frc."""
    ts = pd.to_datetime(frame["ts"])
    minute = ts.dt.hour * 60 + ts.dt.minute
    out = pd.DataFrame(index=frame.index)
    out["minute_sin"] = np.sin(2 * np.pi * minute / 1440)
    out["minute_cos"] = np.cos(2 * np.pi * minute / 1440)
    out["is_weekend"] = np.isin(ts.dt.dayofweek, WEEKEND_DAYS).astype(int)
    out["free_flow_speed_kmh"] = frame["free_flow_speed_kmh"].astype(float).to_numpy()
    out["frc_code"] = frame["frc"].map(frc_code).to_numpy()
    out["point_mean"] = frame["point_id"].map(usual["point"]).fillna(usual["global"]).to_numpy()
    out["point_hour_mean"] = usual_ratio(usual, frame["point_id"], ts.dt.hour, day_type(ts.dt.dayofweek), by_daytype=False)
    return out[TIME_MODEL_FEATURES]


TIME_OF_DAY_METHODS = {
    "time_model_hgb": "Time-of-day model (gradient boosting)",
    "avg_point_hour": "Historical average (point x hour)",
    "avg_point_hour_daytype": "Historical average (point x hour x weekday/weekend)",
}


def time_of_day_ratio(state: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predicted speed ratio for any point/time. state comes from train_masar.py."""
    ts = pd.to_datetime(frame["ts"])
    method = state["method"]
    if method == "time_model_hgb":
        x = time_model_features(frame, state["usual"])
        return np.clip(state["model"].predict(x), 0, 1)
    return usual_ratio(
        state["usual"], frame["point_id"], ts.dt.hour, day_type(ts.dt.dayofweek),
        by_daytype=(method == "avg_point_hour_daytype"),
    )


# ------------------------------------------------- short-term features

LAG_COLUMNS = ["speed_ratio", "current_speed_kmh", "delay_s"]

SHORT_TERM_NUM_FEATURES = (
    [f"{c}_t" for c in LAG_COLUMNS]
    + [f"{c}_lag{k}" for c in LAG_COLUMNS for k in range(1, N_LAGS + 1)]
    + [
        "ratio_roll3_mean", "ratio_roll3_std", "ratio_roll6_mean", "ratio_roll6_std",
        "delay_roll3_mean", "ratio_change", "speed_change", "delay_change",
        "free_flow_speed_kmh", "target_hour_sin", "target_hour_cos",
        "target_dow", "target_is_weekend", "usual_ratio_target",
    ]
)
SHORT_TERM_CAT_FEATURES = ["point_id"]
SHORT_TERM_FEATURES = SHORT_TERM_NUM_FEATURES + SHORT_TERM_CAT_FEATURES


def lag_features(d: pd.DataFrame) -> pd.DataFrame:
    """Features known at reading time t (current + past readings only).

    A lag k counts only if it is really ~k steps back in time; across a
    collection gap it is set to NaN instead of silently using an older reading.
    """
    d = d.sort_values(["point_id", "ts"]).reset_index(drop=True)
    g = d.groupby("point_id", group_keys=False)
    out = d[["point_id", "road", "ts", "free_flow_speed_kmh", "speed_ratio", "congestion_level"]].copy()

    for c in LAG_COLUMNS:
        out[f"{c}_t"] = d[c].astype(float)
    for k in range(1, N_LAGS + 1):
        valid = (d["ts"] - g["ts"].shift(k)) <= pd.Timedelta(minutes=k * STEP_MIN + TOLERANCE_MIN)
        for c in LAG_COLUMNS:
            out[f"{c}_lag{k}"] = g[c].shift(k).astype(float).where(valid)

    ratios = out[["speed_ratio_t"] + [f"speed_ratio_lag{k}" for k in range(1, N_LAGS + 1)]].to_numpy()
    delays = out[["delay_s_t", "delay_s_lag1", "delay_s_lag2"]].to_numpy()
    with warnings.catch_warnings():
        # all-NaN windows (right after a gap) are expected and become NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out["ratio_roll3_mean"] = np.nanmean(ratios[:, :3], axis=1)
        out["ratio_roll3_std"] = np.nanstd(ratios[:, :3], axis=1)
        out["ratio_roll6_mean"] = np.nanmean(ratios[:, :6], axis=1)
        out["ratio_roll6_std"] = np.nanstd(ratios[:, :6], axis=1)
        out["delay_roll3_mean"] = np.nanmean(delays, axis=1)
    out["ratio_change"] = out["speed_ratio_t"] - out["speed_ratio_lag1"]
    out["speed_change"] = out["current_speed_kmh_t"] - out["current_speed_kmh_lag1"]
    out["delay_change"] = out["delay_s_t"] - out["delay_s_lag1"]
    return out


def add_target_calendar(samples: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Calendar features of the time being predicted (t + horizon)."""
    s = samples.copy()
    s["target_time"] = s["ts"] + pd.Timedelta(minutes=horizon_min)
    minute = s["target_time"].dt.hour * 60 + s["target_time"].dt.minute
    s["target_hour"] = s["target_time"].dt.hour
    s["target_hour_sin"] = np.sin(2 * np.pi * minute / 1440)
    s["target_hour_cos"] = np.cos(2 * np.pi * minute / 1440)
    s["target_dow"] = s["target_time"].dt.dayofweek
    s["target_is_weekend"] = np.isin(s["target_dow"], WEEKEND_DAYS).astype(int)
    s["target_daytype"] = day_type(s["target_dow"])
    return s


def horizon_samples(d: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """One row per reading t with the observed level at t + horizon.

    The target is matched by TIME (nearest reading within +/- 5 min of
    t + horizon for the same point), not by row count, so collection gaps can
    never pair t with a reading from much later.
    """
    s = add_target_calendar(lag_features(d), horizon_min)
    future = d[["point_id", "ts", "speed_ratio"]].rename(
        columns={"ts": "target_ts", "speed_ratio": "target_ratio"}
    )
    s = pd.merge_asof(
        s.sort_values("target_time"),
        future.sort_values("target_ts"),
        left_on="target_time",
        right_on="target_ts",
        by="point_id",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=TOLERANCE_MIN),
    )
    s = s.dropna(subset=["target_ts"]).sort_values(["point_id", "ts"]).reset_index(drop=True)
    s["target_level"] = classify_many(s["target_ratio"])
    return s


def latest_samples(d: pd.DataFrame, horizon_min: int) -> pd.DataFrame:
    """Features for the newest reading of each point (for forecasting)."""
    feats = lag_features(d)
    last = feats.loc[feats.groupby("point_id")["ts"].idxmax()]
    return add_target_calendar(last, horizon_min).reset_index(drop=True)


def add_usual(samples: pd.DataFrame, usual: dict) -> pd.DataFrame:
    s = samples.copy()
    s["usual_ratio_target"] = usual_ratio(usual, s["point_id"], s["target_hour"], s["target_daytype"])
    return s
