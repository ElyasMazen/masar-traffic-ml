from __future__ import annotations

from pathlib import Path
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "traffic_training.csv.gz"
MODEL_FILE = BASE_DIR / "models" / "traffic_model.joblib"
METRICS_FILE = BASE_DIR / "data" / "processed" / "model_metrics.json"

df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

split = int(len(df) * 0.80)
train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

global_mean = float(train["traffic_volume"].mean())
road_mean = train.groupby("road_key")["traffic_volume"].mean()
road_hour_mean = train.groupby(["road_key", "hour"])["traffic_volume"].mean()
road_dow_hour_mean = train.groupby(
    ["road_key", "day_of_week", "hour"]
)["traffic_volume"].mean()


def features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["road_mean"] = out["road_key"].map(road_mean).fillna(global_mean)

    idx = pd.MultiIndex.from_frame(out[["road_key", "hour"]])
    out["road_hour_mean"] = pd.Series(
        road_hour_mean.reindex(idx).to_numpy(), index=out.index
    ).fillna(out["road_mean"])

    idx = pd.MultiIndex.from_frame(out[["road_key", "day_of_week", "hour"]])
    out["road_dow_hour_mean"] = pd.Series(
        road_dow_hour_mean.reindex(idx).to_numpy(), index=out.index
    ).fillna(out["road_hour_mean"])

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    return out


feature_names = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "road_mean",
    "road_hour_mean",
    "road_dow_hour_mean",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

train_x = features(train)
test_x = features(test)

model = HistGradientBoostingRegressor(
    max_iter=160,
    learning_rate=0.07,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42,
)
model.fit(train_x[feature_names], train["traffic_volume"])

pred = model.predict(test_x[feature_names])
metrics = {
    "model": "HistGradientBoostingRegressor + historical road encodings",
    "split": "chronological 80/20",
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "train_end": str(train["timestamp"].max()),
    "test_start": str(test["timestamp"].min()),
    "test_end": str(test["timestamp"].max()),
    "MAE": round(float(mean_absolute_error(test["traffic_volume"], pred)), 2),
    "RMSE": round(float(mean_squared_error(test["traffic_volume"], pred) ** 0.5), 2),
    "R2": round(float(r2_score(test["traffic_volume"], pred)), 4),
}

bundle = {
    "model": model,
    "features": feature_names,
    "global_mean": global_mean,
    "road_mean": {str(k): float(v) for k, v in road_mean.items()},
    "road_hour_mean": {
        f"{k[0]}||{int(k[1])}": float(v) for k, v in road_hour_mean.items()
    },
    "road_dow_hour_mean": {
        f"{k[0]}||{int(k[1])}||{int(k[2])}": float(v)
        for k, v in road_dow_hour_mean.items()
    },
    "training_end": str(train["timestamp"].max()),
}

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, MODEL_FILE, compress=3)
METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print("Traffic model trained successfully.")
print(json.dumps(metrics, indent=2))
