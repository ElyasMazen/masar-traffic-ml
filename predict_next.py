import argparse, joblib
from pathlib import Path
import pandas as pd
from masar_features import build_future_features, FEATURES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="models/random_forest_pilot_fullfit.joblib")
    ap.add_argument("--out", default="next_step_forecast.csv")
    args = ap.parse_args()

    history = pd.read_csv(args.csv)
    model = joblib.load(args.model)
    x = build_future_features(history)
    if x.empty:
        raise SystemExit("Not enough recent continuous history (need 3 recent observations per point, gaps <=15 min).")

    pred = model.predict(x[FEATURES])
    proba = model.predict_proba(x[FEATURES])
    classes = model.named_steps["model"].classes_

    result = x[
        ["road","point_id","latest_timestamp_local","forecast_for_local",
         "latest_speed_ratio","latest_level"]
    ].copy()
    result["predicted_level"] = pred
    for i,c in enumerate(classes):
        result[f"prob_{c}"] = proba[:,i]
    result["model_confidence"] = proba.max(axis=1)
    result = result.sort_values(["road","point_id"])
    result.to_csv(args.out,index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
