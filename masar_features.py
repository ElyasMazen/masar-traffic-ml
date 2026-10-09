from pathlib import Path
import pandas as pd
import numpy as np

LABELS = ["Low", "Medium", "High"]
DOW_MAP = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}

NUM_FEATURES = [
    "hour_sin","hour_cos","dow_sin","dow_cos","is_weekend",
    "speed_ratio_lag1","speed_ratio_lag2","speed_ratio_lag3",
    "delay_s_lag1","delay_s_lag2","delay_s_lag3",
    "current_speed_kmh_lag1","current_speed_kmh_lag2","current_speed_kmh_lag3",
    "current_travel_time_s_lag1","free_flow_speed_kmh_lag1",
    "free_flow_travel_time_s_lag1","confidence_lag1","road_closure_lag1",
    "speed_ratio_roll3_mean","speed_ratio_roll3_std",
    "delay_s_roll3_mean","delay_s_roll3_std",
    "current_speed_kmh_roll3_mean","current_speed_kmh_roll3_std"
]
CAT_FEATURES = ["road","point_id","frc","congestion_prev"]
FEATURES = NUM_FEATURES + CAT_FEATURES

def engineer_features(path_or_df):
    if isinstance(path_or_df, (str, Path)):
        d = pd.read_csv(path_or_df)
    else:
        d = path_or_df.copy()

    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"])
    d["timestamp_local"] = pd.to_datetime(d["timestamp_local"])
    d = d.sort_values(["point_id","timestamp_local"]).reset_index(drop=True)

    minute_of_day = d["timestamp_local"].dt.hour*60 + d["timestamp_local"].dt.minute + d["timestamp_local"].dt.second/60
    d["hour_sin"] = np.sin(2*np.pi*minute_of_day/1440)
    d["hour_cos"] = np.cos(2*np.pi*minute_of_day/1440)
    d["dow_num"] = d["day_of_week"].map(DOW_MAP)
    d["dow_sin"] = np.sin(2*np.pi*d["dow_num"]/7)
    d["dow_cos"] = np.cos(2*np.pi*d["dow_num"]/7)

    g = d.groupby("point_id", group_keys=False)
    d["gap_min"] = g["timestamp_local"].diff().dt.total_seconds()/60

    lag_cols = [
        "speed_ratio","delay_s","current_speed_kmh","current_travel_time_s",
        "free_flow_speed_kmh","free_flow_travel_time_s","confidence","road_closure"
    ]
    for col in lag_cols:
        for lag in [1,2,3]:
            d[f"{col}_lag{lag}"] = g[col].shift(lag)

    d["congestion_prev"] = g["congestion_level"].shift(1)

    for col in ["speed_ratio","delay_s","current_speed_kmh"]:
        shifted = g[col].shift(1)
        d[f"{col}_roll3_mean"] = shifted.groupby(d["point_id"]).rolling(3).mean().reset_index(level=0, drop=True)
        d[f"{col}_roll3_std"] = shifted.groupby(d["point_id"]).rolling(3).std(ddof=0).reset_index(level=0, drop=True)

    d["gap_prev1"] = g["gap_min"].shift(1)
    d["gap_prev2"] = g["gap_min"].shift(2)

    continuity = (
        d[["gap_min","gap_prev1","gap_prev2"]].le(15).all(axis=1)
        & d[["gap_min","gap_prev1","gap_prev2"]].gt(0).all(axis=1)
    )
    return d, d.loc[continuity].copy()

def build_future_features(history):
    h = history.copy()
    h["timestamp_local"] = pd.to_datetime(h["timestamp_local"])
    h = h.sort_values(["point_id","timestamp_local"])
    rows = []

    for pid, gp in h.groupby("point_id"):
        gp = gp.sort_values("timestamp_local")
        last3 = gp.tail(3)
        if len(last3) < 3:
            continue

        gaps = last3["timestamp_local"].diff().dt.total_seconds().div(60).dropna()
        if (gaps > 15).any():
            continue

        last = last3.iloc[-1]
        future_ts = last["timestamp_local"] + pd.Timedelta(minutes=10)
        minute = future_ts.hour*60 + future_ts.minute + future_ts.second/60
        dow = future_ts.dayofweek

        row = {
            "road":last["road"], "point_id":pid, "frc":last["frc"],
            "congestion_prev":last["congestion_level"],
            "hour_sin":np.sin(2*np.pi*minute/1440),
            "hour_cos":np.cos(2*np.pi*minute/1440),
            "dow_sin":np.sin(2*np.pi*dow/7),
            "dow_cos":np.cos(2*np.pi*dow/7),
            "is_weekend":int(dow in [4,5]),
        }

        newest_first = last3.iloc[::-1]
        for col in ["speed_ratio","delay_s","current_speed_kmh"]:
            for i,v in enumerate(newest_first[col].values, start=1):
                row[f"{col}_lag{i}"] = v

        for col in [
            "current_travel_time_s","free_flow_speed_kmh",
            "free_flow_travel_time_s","confidence","road_closure"
        ]:
            row[f"{col}_lag1"] = last[col]

        for col in ["speed_ratio","delay_s","current_speed_kmh"]:
            row[f"{col}_roll3_mean"] = last3[col].mean()
            row[f"{col}_roll3_std"] = last3[col].std(ddof=0)

        row["latest_timestamp_local"] = last["timestamp_local"]
        row["forecast_for_local"] = future_ts
        row["latest_speed_ratio"] = last["speed_ratio"]
        row["latest_level"] = last["congestion_level"]
        rows.append(row)

    return pd.DataFrame(rows)
