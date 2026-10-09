import argparse, json, joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from masar_features import engineer_features, FEATURES, NUM_FEATURES, CAT_FEATURES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Downloaded TomTom CSV snapshot. Never point this at the live GitHub workflow.")
    ap.add_argument("--out", default="retrain_outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models_dir = out/"models"
    models_dir.mkdir(exist_ok=True)

    _, data = engineer_features(args.csv)
    ts = np.array(sorted(data["timestamp_local"].unique()))
    i1 = int(np.floor(len(ts)*0.72))
    i2 = int(np.floor(len(ts)*0.80))
    train = data[data["timestamp_local"].isin(set(ts[:i1]))]
    val = data[data["timestamp_local"].isin(set(ts[i1:i2]))]
    test = data[data["timestamp_local"].isin(set(ts[i2:]))]
    dev = pd.concat([train,val]).sort_values("timestamp_local")

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8,min_samples_leaf=10,class_weight="balanced",random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,max_depth=12,min_samples_leaf=5,
            class_weight="balanced_subsample",random_state=42,n_jobs=-1
        )
    }

    val_rows = []
    for name, est in candidates.items():
        pipe = Pipeline([("preprocess",clone(pre)),("model",clone(est))])
        pipe.fit(train[FEATURES], train["congestion_level"])
        p = pipe.predict(val[FEATURES])
        val_rows.append({
            "model":name,
            "accuracy":accuracy_score(val["congestion_level"],p),
            "balanced_accuracy":balanced_accuracy_score(val["congestion_level"],p),
            "macro_f1":f1_score(val["congestion_level"],p,average="macro")
        })

    val_metrics = pd.DataFrame(val_rows).sort_values("macro_f1",ascending=False)
    val_metrics.to_csv(out/"validation_metrics.csv",index=False)
    best = val_metrics.iloc[0]["model"]

    evaluated = Pipeline([("preprocess",clone(pre)),("model",clone(candidates[best]))])
    evaluated.fit(dev[FEATURES],dev["congestion_level"])
    pred = evaluated.predict(test[FEATURES])
    pd.DataFrame([{
        "selected_model":best,
        "accuracy":accuracy_score(test["congestion_level"],pred),
        "balanced_accuracy":balanced_accuracy_score(test["congestion_level"],pred),
        "macro_f1":f1_score(test["congestion_level"],pred,average="macro")
    }]).to_csv(out/"selected_test_metrics.csv",index=False)
    joblib.dump(evaluated,models_dir/"evaluated_model.joblib")

    # Fit a separate demo model on all eligible pilot rows AFTER evaluation.
    demo = Pipeline([("preprocess",clone(pre)),("model",clone(candidates[best]))])
    demo.fit(data[FEATURES],data["congestion_level"])
    joblib.dump(demo,models_dir/"pilot_fullfit_model.joblib")

    print(val_metrics.to_string(index=False))
    print(f"\nSelected on validation Macro-F1: {best}")
    print(f"Saved to: {out.resolve()}")

if __name__ == "__main__":
    main()
