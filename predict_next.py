"""
Forecast the congestion level +h min after the newest reading of each point.
The data is resampled to the step the model was trained with (step 30: +30/+60;
step 10: +10/+30/+60).

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
    ap.add_argument("--horizon", type=int, choices=mf.ALL_HORIZONS_MIN, default=30)
    ap.add_argument("--csv", default=None, help="Local snapshot CSV. Default: the raw GitHub CSV.")
    ap.add_argument("--model", default=None, help="Default: models/short_term_rf_<horizon>min.joblib")
    ap.add_argument("--out", default="next_step_forecast.csv")
    args = ap.parse_args()

    path = Path(args.model or ROOT / "models" / f"short_term_rf_{args.horizon}min.joblib")
    if not path.exists():
        raise SystemExit(f"{path} not found - horizons trained at the default step are {mf.HORIZONS_MIN}.")
    bundle = joblib.load(path)
    mf.set_step(bundle.get("step_min", 10))   # bundles from before the step setting were 10-min
    d, info = mf.load_data(args.csv)
    x = mf.add_usual(mf.latest_samples(d, bundle["horizon_min"]), bundle["usual"])

    model = bundle["pipeline"]
    proba = model.predict_proba(x[bundle["features"]])
    result = x[["road", "point_id", "reading_ts", "ts", "target_time", "speed_ratio_t", "congestion_level"]].rename(columns={
        "reading_ts": "latest_reading", "ts": "latest_slot", "target_time": "forecast_for",
        "speed_ratio_t": "latest_speed_ratio", "congestion_level": "latest_level",
    })
    result["persistence_level"] = result["latest_level"]
    result["predicted_level"] = model.classes_[proba.argmax(axis=1)]
    for i, c in enumerate(model.classes_):
        result[f"prob_{c}"] = proba[:, i].round(4)
    result["model_probability"] = proba.max(axis=1).round(4)
    result = result.sort_values(["road", "point_id"])
    result.to_csv(args.out, index=False)

    print(f"Data: {info['source']} (latest reading {info['last_reading_time']}, "
          f"{info['last_reading']} slot on the {mf.STEP_MIN}-min grid)")
    print(f"Model trained through {bundle['trained_through']}; step {mf.STEP_MIN} min; "
          f"horizon +{bundle['horizon_min']} min\n")
    print(result.to_string(index=False))
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
