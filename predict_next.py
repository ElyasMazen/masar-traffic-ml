"""
Forecast the congestion level +10/+30/+60 min after the newest reading of each point.

    python predict_next.py --horizon 30                     # raw GitHub CSV
    python predict_next.py --horizon 30 --csv snapshot.csv  # a local snapshot

Never calls the TomTom API.
"""

import argparse
from pathlib import Path

import joblib

import masar_features as mf

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, choices=mf.HORIZONS_MIN, default=30)
    ap.add_argument("--csv", default=None, help="Local snapshot CSV. Default: the raw GitHub CSV.")
    ap.add_argument("--model", default=None, help="Default: models/short_term_rf_<horizon>min.joblib")
    ap.add_argument("--out", default="next_step_forecast.csv")
    args = ap.parse_args()

    bundle = joblib.load(args.model or ROOT / "models" / f"short_term_rf_{args.horizon}min.joblib")
    d, info = mf.load_data(args.csv)
    x = mf.add_usual(mf.latest_samples(d, bundle["horizon_min"]), bundle["usual"])

    model = bundle["pipeline"]
    proba = model.predict_proba(x[bundle["features"]])
    result = x[["road", "point_id", "ts", "target_time", "speed_ratio_t", "congestion_level"]].rename(columns={
        "ts": "latest_reading", "target_time": "forecast_for",
        "speed_ratio_t": "latest_speed_ratio", "congestion_level": "latest_level",
    })
    result["persistence_level"] = result["latest_level"]
    result["predicted_level"] = model.classes_[proba.argmax(axis=1)]
    for i, c in enumerate(model.classes_):
        result[f"prob_{c}"] = proba[:, i].round(4)
    result["model_probability"] = proba.max(axis=1).round(4)
    result = result.sort_values(["road", "point_id"])
    result.to_csv(args.out, index=False)

    print(f"Data: {info['source']} (latest reading {info['last_reading']})")
    print(f"Model trained through {bundle['trained_through']}; horizon +{bundle['horizon_min']} min\n")
    print(result.to_string(index=False))
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
