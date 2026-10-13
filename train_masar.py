"""
Masar pilot training and evaluation.

    python train_masar.py                       # raw GitHub CSV, saved to data/snapshots/
    python train_masar.py --csv snapshot.csv    # a local snapshot instead

Reading the raw GitHub CSV is read-only. Never call the TomTom API and never
write to the data repo.

Evaluates, on chronological splits only (train on earlier time, test later):
  A. short-term congestion level at +10/+30/+60 min:
     persistence vs historical average vs Random Forest (class_weight="balanced")
  B. time-of-day speed for any date/hour (cards, map, 12-hour chart):
     gradient-boosting time model vs point x hour vs point x hour x weekday/weekend average

Writes outputs/, charts/ and models/ (the web prototype reads models/ and
outputs/results.json).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

import masar_features as mf  # noqa: E402

ROOT = Path(__file__).resolve().parent
LABELS = mf.LABELS
FEATURES = mf.SHORT_TERM_FEATURES

SHORT_METHODS = {
    "persistence": "Persistence (next level = current level)",
    "historical": "Historical average (point x hour x weekday/weekend)",
    "random_forest": "Random Forest (class_weight=balanced)",
}
# Differences smaller than this on one or two test days are treated as noise.
F1_NOISE = 0.01


# ---------------------------------------------------------------- models

def make_rf() -> Pipeline:
    pre = ColumnTransformer(
        [("point", OneHotEncoder(handle_unknown="ignore", sparse_output=False), mf.SHORT_TERM_CAT_FEATURES)],
        remainder="passthrough",
        sparse_threshold=0,
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", rf)])


def make_time_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42,
    )


# ---------------------------------------------------------------- splits

def define_splits(d: pd.DataFrame) -> list[dict]:
    ts = np.sort(d["ts"].unique())
    cut = pd.Timestamp(ts[int(len(ts) * 0.80)])
    splits = [{
        "name": "latest",
        "label": "Latest 20% of time",
        "train_end": cut,
        "test_start": cut,
        "test_end": d["ts"].max() + pd.Timedelta(seconds=1),
    }]

    # Main test: hold out the last 3+ days, starting at midnight of the latest
    # weekend day that still leaves 3 days of test data, so the test holds both
    # weekdays and a weekend day. Train on everything before it.
    first, last = d["ts"].min(), d["ts"].max()
    days = pd.date_range(first.normalize() + pd.Timedelta(days=1), last.normalize(), freq="D")
    starts = [day for day in days if day.dayofweek in mf.WEEKEND_DAYS and last - day >= pd.Timedelta(days=3)]
    start = max(starts) if starts else last.normalize() - pd.Timedelta(days=3)
    if start > first:
        n_days = (last.normalize() - start).days + 1
        splits.insert(0, {
            "name": "holdout",
            "label": f"Last {n_days} days ({start:%a %d %b} - {last:%a %d %b})",
            "train_end": start,
            "test_start": start,
            "test_end": last + pd.Timedelta(seconds=1),
        })
    return splits


def readings_split(d: pd.DataFrame, sp: dict):
    train = d[d["ts"] < sp["train_end"]]
    test = d[(d["ts"] >= sp["test_start"]) & (d["ts"] < sp["test_end"])]
    return train, test


def samples_split(s: pd.DataFrame, sp: dict):
    # Train rows must have their TARGET before the cut, so nothing from the
    # test period leaks in through t + horizon.
    train = s[s["target_ts"] < sp["train_end"]]
    test = s[(s["ts"] >= sp["test_start"]) & (s["ts"] < sp["test_end"]) & (s["target_ts"] < sp["test_end"])]
    return train, test


def subsets(daytypes: pd.Series):
    """'all', plus weekday/weekend separately when the test set has both."""
    yield "all", np.ones(len(daytypes), dtype=bool)
    present = sorted(set(daytypes))
    if len(present) > 1:
        for t in present:
            yield t, (daytypes == t).to_numpy()


# ---------------------------------------------------------------- metrics

def scores(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [l for l in LABELS if (y_true == l).any() or (y_pred == l).any()]
    row = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
    }
    for l in LABELS:
        support = int((y_true == l).sum())
        row[f"support_{l}"] = support
        row[f"recall_{l}"] = float(((y_pred == l) & (y_true == l)).sum() / support) if support else np.nan
    return row


# --------------------------------------------------- A. short-term models

def evaluate_short_term(d, splits):
    rows, cms = [], []
    samples = {h: mf.horizon_samples(d, h) for h in mf.HORIZONS_MIN}
    for sp in splits:
        usual = mf.fit_usual(readings_split(d, sp)[0])
        for h, s in samples.items():
            train, test = samples_split(mf.add_usual(s, usual), sp)
            model = make_rf().fit(train[FEATURES], train["target_level"])
            preds = {
                "persistence": test["congestion_level"].to_numpy(),
                "historical": mf.classify_many(test["usual_ratio_target"]),
                "random_forest": model.predict(test[FEATURES]),
            }
            test_daytypes = "+".join(sorted(set(test["target_daytype"])))
            for method, p in preds.items():
                for subset, mask in subsets(test["target_daytype"]):
                    rows.append({
                        "split": sp["name"], "horizon_min": h, "method": method, "subset": subset,
                        "test_daytypes": test_daytypes, "train_n": int(len(train)),
                        **scores(test["target_level"].to_numpy()[mask], p[mask]),
                    })
                cm = confusion_matrix(test["target_level"], p, labels=LABELS)
                for i, t in enumerate(LABELS):
                    for j, q in enumerate(LABELS):
                        cms.append({"split": sp["name"], "horizon_min": h, "method": method,
                                    "true": t, "pred": q, "count": int(cm[i, j])})
    return pd.DataFrame(rows), pd.DataFrame(cms)


def choose_horizon(results: pd.DataFrame, splits: list[dict]) -> dict:
    names = [sp["name"] for sp in splits]
    primary = "holdout" if "holdout" in names else "latest"
    overall = results[results["subset"] == "all"]

    def metric(split, h, method, name):
        r = overall[(overall["split"] == split) & (overall["horizon_min"] == h) & (overall["method"] == method)]
        return float(r[name].iloc[0])

    def f1(split, h, method):
        return metric(split, h, method, "macro_f1")

    labels = {sp["name"]: sp["label"] for sp in splits}
    comparison = [
        {"split": s, "split_label": labels[s], "horizon_min": h,
         "random_forest_macro_f1": f1(s, h, "random_forest"), "persistence_macro_f1": f1(s, h, "persistence")}
        for s in names for h in mf.HORIZONS_MIN
    ]
    margin = {(c["split"], c["horizon_min"]): c["random_forest_macro_f1"] - c["persistence_macro_f1"] for c in comparison}

    h = max(mf.HORIZONS_MIN, key=lambda x: margin[(primary, x)])
    others = [s for s in names if s != primary]
    # "Beats" needs a margin above noise on the primary test AND no loss on the others.
    if margin[(primary, h)] > F1_NOISE and all(margin[(s, h)] > 0 for s in others):
        verdict, beats = "Overall, the model beats persistence at this horizon, by a small margin, on every test set.", True
    elif margin[(primary, h)] >= -F1_NOISE:
        verdict, beats = "That is within noise: the model does not clearly beat persistence.", False
    else:
        verdict, beats = "The model does not beat persistence: \"next level = current level\" is better.", False

    rf_high = metric(primary, h, "random_forest", "recall_High")
    pe_high = metric(primary, h, "persistence", "recall_High")
    if not (math.isnan(rf_high) or math.isnan(pe_high)) and rf_high < pe_high:
        verdict += (f" But it catches fewer High-congestion readings than persistence"
                    f" ({rf_high:.0%} vs {pe_high:.0%} High recall).")

    # The overall score can hide a day type where the model is behind.
    for subset in ("weekday", "weekend"):
        part = results[(results["split"] == primary) & (results["horizon_min"] == h) & (results["subset"] == subset)]
        if part.empty:
            continue
        rf_f1 = float(part.loc[part["method"] == "random_forest", "macro_f1"].iloc[0])
        pe_f1 = float(part.loc[part["method"] == "persistence", "macro_f1"].iloc[0])
        if rf_f1 < pe_f1:
            verdict += (f" On the {subset} part of the test it is behind persistence"
                        f" ({rf_f1:.3f} vs {pe_f1:.3f}).")

    losing = [x for x in mf.HORIZONS_MIN if margin[(primary, x)] <= F1_NOISE]
    if losing:
        verdict += f" At +{' and +'.join(str(x) for x in losing)} min it does not beat persistence."

    def pair(s):
        c = next(c for c in comparison if c["split"] == s and c["horizon_min"] == h)
        return f"{c['random_forest_macro_f1']:.3f} vs persistence {c['persistence_macro_f1']:.3f} ({labels[s]})"

    sentence = f"At +{h} min: model F1 " + "; ".join(pair(s) for s in [primary] + others) + f". {verdict}"
    return {
        "primary_split": primary,
        "horizon_min": h,
        "random_forest_macro_f1": f1(primary, h, "random_forest"),
        "persistence_macro_f1": f1(primary, h, "persistence"),
        "margin": margin[(primary, h)],
        "beats_persistence": beats,
        "rule": f"horizon with the largest Random Forest - persistence macro-F1 margin on '{labels[primary]}'",
        "sentence": sentence,
        "comparison": comparison,
    }


# --------------------------------------------------- B. time-of-day methods

def time_of_day_states(train: pd.DataFrame) -> dict:
    usual = mf.fit_usual(train)
    hgb = make_time_model().fit(mf.time_model_features(train, usual), train["speed_ratio"])
    return {
        "time_model_hgb": {"method": "time_model_hgb", "usual": usual, "model": hgb},
        "avg_point_hour": {"method": "avg_point_hour", "usual": usual, "model": None},
        "avg_point_hour_daytype": {"method": "avg_point_hour_daytype", "usual": usual, "model": None},
    }


def evaluate_time_of_day(d, splits) -> pd.DataFrame:
    rows = []
    for sp in splits:
        train, test = readings_split(d, sp)
        test_daytypes = "+".join(sorted(set(test["daytype"])))
        for method, state in time_of_day_states(train).items():
            ratio = mf.time_of_day_ratio(state, test)
            for subset, mask in subsets(test["daytype"]):
                t, r = test[mask], ratio[mask]
                rows.append({
                    "split": sp["name"], "method": method, "subset": subset, "test_daytypes": test_daytypes,
                    "mae_speed_kmh": float(mean_absolute_error(t["current_speed_kmh"], r * t["free_flow_speed_kmh"])),
                    "mae_speed_ratio": float(mean_absolute_error(t["speed_ratio"], r)),
                    **scores(t["congestion_level"], mf.classify_many(r)),
                })
    return pd.DataFrame(rows)


def choose_time_method(results: pd.DataFrame, primary: str) -> dict:
    r = results[(results["split"] == primary) & (results["subset"] == "all")].sort_values("mae_speed_kmh")
    best = r.iloc[0]
    label = mf.TIME_OF_DAY_METHODS[best["method"]]
    if best["method"] != "time_model_hgb":
        label = label.replace("Historical average", "Historical average (pilot)")
    return {
        "method": best["method"],
        "label": label,
        "rule": f"lowest speed error (MAE km/h) on the '{primary}' split",
        "mae_speed_kmh": float(best["mae_speed_kmh"]),
        "accuracy": float(best["accuracy"]),
    }


# ---------------------------------------------------------------- audit

def audit(d: pd.DataFrame, out: Path) -> dict:
    runs = pd.Series(np.sort(d["ts"].unique()))
    gaps = runs.diff().dt.total_seconds().div(60)
    big = pd.DataFrame({"gap_start": runs.shift(1), "gap_end": runs, "gap_min": gaps.round(2)})
    big = big[big["gap_min"] > 15]
    big.to_csv(out / "gaps_over_15min.csv", index=False)
    d[d["road_closure"].astype(str).str.lower() == "true"].to_csv(out / "road_closure_rows_for_review.csv", index=False)
    dist = d["congestion_level"].value_counts().reindex(LABELS).fillna(0).astype(int)
    dist.rename("rows").to_frame().assign(percent=lambda x: (100 * x["rows"] / x["rows"].sum()).round(2)) \
        .to_csv(out / "label_distribution.csv", index_label="congestion_level")
    pd.crosstab(d["point_id"], d["congestion_level"], normalize="index").reindex(columns=LABELS).mul(100).round(2) \
        .to_csv(out / "label_distribution_by_point_percent.csv")
    by_point = pd.crosstab(d["point_id"], d["congestion_level"]).reindex(columns=LABELS).fillna(0).astype(int)
    by_point.to_csv(out / "label_counts_by_point.csv")
    day_list = pd.Series(d["ts"].dt.normalize().unique())
    weekend = day_list.dt.dayofweek.isin(mf.WEEKEND_DAYS)
    result = {
        "rows": int(len(d)),
        "runs": int(len(runs)),
        "points": int(d["point_id"].nunique()),
        "roads": int(d["road"].nunique()),
        "first_reading": str(d["ts"].min()),
        "last_reading": str(d["ts"].max()),
        "readings_per_day": {str(k.date()): int(v) for k, v in d.groupby(d["ts"].dt.normalize()).size().items()},
        "gaps_over_15min": int(len(big)),
        "max_gap_min": float(gaps.max()),
        "road_closure_rows": int((d["road_closure"].astype(str).str.lower() == "true").sum()),
        "calendar_days": int(len(day_list)),
        "weekdays": [f"{x:%a %d %b}" for x in day_list[~weekend]],
        "weekend_days": [f"{x:%a %d %b}" for x in day_list[weekend]],
        "label_counts": dist.to_dict(),
        "label_distribution_percent": (100 * dist / dist.sum()).round(2).to_dict(),
        "high_by_point": by_point["High"].to_dict(),
    }
    (out / "data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------- charts

METHOD_COLORS = {"persistence": "#8a96a3", "historical": "#d6a13b", "random_forest": "#0d5a4e"}


def chart_short_term(results: pd.DataFrame, splits, metric: str, title: str, path: Path):
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4.2), squeeze=False)
    overall = results[results["subset"] == "all"]
    width = 0.26
    for ax, sp in zip(axes[0], splits):
        r = overall[overall["split"] == sp["name"]]
        x = np.arange(len(mf.HORIZONS_MIN))
        for i, method in enumerate(SHORT_METHODS):
            vals = [r[(r["horizon_min"] == h) & (r["method"] == method)][metric].iloc[0] for h in mf.HORIZONS_MIN]
            bars = ax.bar(x + (i - 1) * width, np.nan_to_num(vals), width, label=SHORT_METHODS[method],
                          color=METHOD_COLORS[method])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, (0 if math.isnan(v) else v) + 0.01,
                        "n/a" if math.isnan(v) else f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x, [f"+{h} min" for h in mf.HORIZONS_MIN])
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{sp['label']}\n(test: {r['test_daytypes'].iloc[0]})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(title)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=8, loc="lower center", ncol=len(SHORT_METHODS), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_confusions(cms: pd.DataFrame, sp: dict, h: int, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, method in zip(axes, SHORT_METHODS):
        c = cms[(cms["split"] == sp["name"]) & (cms["horizon_min"] == h) & (cms["method"] == method)]
        m = c.pivot(index="true", columns="pred", values="count").reindex(index=LABELS, columns=LABELS)
        ax.imshow(m.to_numpy(), cmap="Greens")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(m.iat[i, j]), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("predicted")
        ax.set_title(SHORT_METHODS[method], fontsize=8)
    axes[0].set_ylabel("actual")
    fig.suptitle(f"+{h} min · {sp['label']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_time_of_day(results: pd.DataFrame, splits, path: Path):
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    colors = ["#0d5a4e", "#8a96a3", "#d6a13b"]
    for ax, sp in zip(axes[0], splits):
        r = results[(results["split"] == sp["name"]) & (results["subset"] == "all")]
        names = [mf.TIME_OF_DAY_METHODS[m].replace("Historical average", "Hist. avg") for m in r["method"]]
        bars = ax.barh(names, r["mae_speed_kmh"], color=colors[: len(r)])
        for b, (mae, acc) in zip(bars, zip(r["mae_speed_kmh"], r["accuracy"])):
            ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2, f"±{mae:.1f} km/h · acc {acc:.0%}",
                    va="center", fontsize=8)
        ax.set_xlabel("speed error (MAE, km/h) - lower is better")
        ax.set_title(f"{sp['label']} (test: {r['test_daytypes'].iloc[0]})", fontsize=10)
        ax.set_xlim(0, r["mae_speed_kmh"].max() * 1.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Time-of-day prediction: model vs historical averages")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_data(d: pd.DataFrame, charts: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    for road, g in d.groupby("road"):
        s = g.set_index("ts")["speed_ratio"].resample("30min").mean()
        ax.plot(s.index, s.to_numpy(), label=road, linewidth=1.2)
    for day in pd.date_range(d["ts"].min().normalize(), d["ts"].max().normalize(), freq="D"):
        if day.dayofweek in mf.WEEKEND_DAYS:
            ax.axvspan(day, day + pd.Timedelta(days=1), color="#d6a13b", alpha=0.12)
    ax.axhline(mf.LOW_MIN_RATIO, color="#999", linestyle=":", linewidth=1)
    ax.axhline(mf.MEDIUM_MIN_RATIO, color="#999", linestyle=":", linewidth=1)
    ax.set_ylabel("speed ratio (30-min mean)")
    ax.set_title("Speed ratio by road (shaded = Fri/Sat weekend)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(charts / "road_speed_ratio_over_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    dist = d["congestion_level"].value_counts().reindex(LABELS)
    ax.bar(LABELS, dist.to_numpy(), color=["#269b72", "#d6a13b", "#d64f49"])
    ax.set_title("Congestion level distribution (all readings)")
    fig.tight_layout()
    fig.savefig(charts / "label_distribution.png", dpi=150)
    plt.close(fig)


def chart_importance(model: Pipeline, h: int, out: Path, charts: Path):
    names = [n.split("__", 1)[1] for n in model.named_steps["preprocess"].get_feature_names_out()]
    imp = pd.Series(model.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
    imp.rename("importance").to_csv(out / "random_forest_feature_importance.csv", index_label="feature")
    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.to_numpy(), color="#0d5a4e")
    ax.set_title(f"Random Forest +{h} min - top 20 features (impurity importance)")
    fig.tight_layout()
    fig.savefig(charts / "random_forest_feature_importance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def clean(value):
    """JSON-safe: NaN -> None, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Local snapshot CSV. Default: the raw GitHub CSV.")
    ap.add_argument("--snapshot-dir", default=str(ROOT / "data" / "snapshots"),
                    help="Where downloaded copies are saved as dated snapshots.")
    ap.add_argument("--out", default=str(ROOT), help="Folder that receives outputs/, charts/ and models/.")
    args = ap.parse_args()

    base = Path(args.out)
    out, charts, models = base / "outputs", base / "charts", base / "models"
    for folder in (out, charts, models):
        folder.mkdir(parents=True, exist_ok=True)

    d, info = mf.load_data(args.csv, args.snapshot_dir)
    for key in ("source", "snapshot"):
        # Store repo-relative paths so results.json carries no machine-specific paths.
        if info[key] and Path(info[key]).is_absolute() and Path(info[key]).is_relative_to(ROOT):
            info[key] = Path(info[key]).relative_to(ROOT).as_posix()
    print(f"Loaded {info['rows']:,} readings from {info['source']}")
    print(f"  {info['first_reading']} -> {info['last_reading']} ({info['days']} calendar days)")
    if info["snapshot"] and args.csv is None:
        print(f"  saved snapshot: {info['snapshot']}")

    audit_info = audit(d, out)
    splits = define_splits(d)
    for sp in splits:
        print(f"Split '{sp['name']}': train < {sp['train_end']}, test {sp['test_start']} -> {sp['test_end']}")

    # A. short-term
    short, cms = evaluate_short_term(d, splits)
    short.to_csv(out / "short_term_results.csv", index=False)
    cms.to_csv(out / "confusion_matrices.csv", index=False)
    selected_h = choose_horizon(short, splits)

    # B. time-of-day
    tod = evaluate_time_of_day(d, splits)
    tod.to_csv(out / "time_of_day_results.csv", index=False)
    selected_t = choose_time_method(tod, selected_h["primary_split"])

    split_rows = []
    for sp in splits:
        train, test = readings_split(d, sp)
        split_rows.append({
            "split": sp["name"], "label": sp["label"],
            "train_start": str(train["ts"].min()), "train_end": str(train["ts"].max()),
            "test_start": str(test["ts"].min()), "test_end": str(test["ts"].max()),
            "train_readings": int(len(train)), "test_readings": int(len(test)),
            "test_daytypes": "+".join(sorted(set(test["daytype"]))),
        })
    pd.DataFrame(split_rows).to_csv(out / "split_summary.csv", index=False)
    primary_row = next(r for r in split_rows if r["split"] == selected_h["primary_split"])

    # Charts
    chart_short_term(short, splits, "macro_f1", "Macro-F1 by horizon", charts / "short_term_macro_f1.png")
    chart_short_term(short, splits, "recall_High", "High-congestion recall by horizon", charts / "short_term_high_recall.png")
    for sp in splits:
        for h in mf.HORIZONS_MIN:
            chart_confusions(cms, sp, h, charts / f"confusion_{sp['name']}_{h}min.png")
    chart_time_of_day(tod, splits, charts / "time_of_day_comparison.png")
    chart_data(d, charts)

    # Final models for the app: refit on every reading after evaluation.
    usual_all = mf.fit_usual(d)
    latest_rows = []
    for h in mf.HORIZONS_MIN:
        s = mf.add_usual(mf.horizon_samples(d, h), usual_all)
        model = make_rf().fit(s[FEATURES], s["target_level"])
        joblib.dump({
            "pipeline": model, "features": FEATURES, "horizon_min": h,
            "usual": usual_all, "trained_from": info["first_reading"], "trained_through": info["last_reading"],
        }, models / f"short_term_rf_{h}min.joblib", compress=3)
        if h == selected_h["horizon_min"]:
            chart_importance(model, h, out, charts)

        latest = mf.add_usual(mf.latest_samples(d, h), usual_all)
        proba = model.predict_proba(latest[FEATURES])
        for i, row in enumerate(latest.itertuples(index=False)):
            latest_rows.append({
                "horizon_min": h, "road": row.road, "point_id": row.point_id,
                "latest_reading": row.ts, "forecast_for": row.target_time,
                "latest_level": row.congestion_level, "predicted_level": model.classes_[proba[i].argmax()],
                **{f"prob_{c}": round(float(p), 4) for c, p in zip(model.classes_, proba[i])},
            })
    pd.DataFrame(latest_rows).to_csv(out / "latest_forecast.csv", index=False)

    final_states = time_of_day_states(d)
    joblib.dump({
        **final_states[selected_t["method"]],
        "label": selected_t["label"],
        "trained_from": info["first_reading"],
        "trained_through": info["last_reading"],
    }, models / "time_of_day.joblib", compress=3)

    days = pd.Series(d["ts"].dt.normalize().unique())
    results = clean({
        "data": {
            **info,
            "day_names": [f"{day:%a %d %b}" for day in days],
            "weekdays_with_data": sorted({int(x) for x in d["dow"].unique()}),
            "trained_through": info["last_reading"],
            "audit": audit_info,
        },
        # Two kinds of model: the evaluated ones (fitted on each split's train
        # part, scored on its test part only) and the full-data ones the app
        # predicts with (refitted on every reading after evaluation; no score).
        "models": {
            "evaluated": {
                "split": selected_h["primary_split"],
                "train_start": primary_row["train_start"], "train_end": primary_row["train_end"],
                "test_start": primary_row["test_start"], "test_end": primary_row["test_end"],
                "note": "All reported metrics come from these models on the held-out test period only.",
            },
            "final": {
                "train_start": info["first_reading"], "train_end": info["last_reading"], "rows": info["rows"],
                "note": "Refitted on all readings (train + test) after evaluation; used for every prediction in the app.",
            },
        },
        "thresholds": {"low_min_ratio": mf.LOW_MIN_RATIO, "medium_min_ratio": mf.MEDIUM_MIN_RATIO},
        "splits": split_rows,
        "short_term": {
            "methods": SHORT_METHODS,
            "selected": selected_h,
            "results": short.to_dict(orient="records"),
        },
        "time_of_day": {
            "methods": mf.TIME_OF_DAY_METHODS,
            "selected": selected_t,
            "results": tod.to_dict(orient="records"),
        },
    })
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    show = short[short["subset"] == "all"].pivot_table(
        index=["split", "horizon_min"], columns="method", values="macro_f1"
    ).round(3)
    print("\nShort-term macro-F1 (all test rows):")
    print(show.to_string())
    print("\nTime-of-day (all test rows):")
    print(tod[tod["subset"] == "all"][["split", "method", "mae_speed_kmh", "accuracy", "macro_f1", "recall_High"]]
          .round(3).to_string(index=False))
    print(f"\nSelected horizon: {selected_h['sentence']}")
    print(f"Selected time-of-day method: {selected_t['label']} ({selected_t['rule']})")
    print(f"Outputs written to {base.resolve()}")


if __name__ == "__main__":
    main()
