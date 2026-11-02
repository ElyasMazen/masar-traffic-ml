"""
Build and check the Masar SAFETY RISK INDICATOR offline (not accident prediction).

    python safety_risk.py                                   # raw 10-min snapshot, defaults
    python safety_risk.py --csv <snapshot> --step 10 --window 60

Writes:
    outputs/safety_risk.csv          per reading: components, scores, index, both bands, main reason
    outputs/safety_validation.json   behaviour, overlap with congestion, sensitivity,
                                     Isolation Forest cross-check, incident proxy, top moments,
                                     10-min vs 30-min comparison
    charts/safety_*.png

There is no crash data: every check here is a sanity / robustness check, not
validation against accidents. Reads the data repo only; never calls the TomTom API.
Does not touch the congestion models or their results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402

import masar_features as mf  # noqa: E402
import masar_safety as ms  # noqa: E402

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "snapshots" / "jeddah_tomtom_flow_20261024_145800.csv"
DEFAULT_INCIDENTS = ROOT.parent / "tomtomdata" / "jeddah_incidents.csv"
BANDS = ["Low", "Medium", "High"]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def load(csv: Path, step: int):
    mf.set_step(step)
    readings, info = mf.load_data(str(csv))
    return readings, info


def high_set(index: pd.Series, top: float = 0.10) -> set:
    """Readings in the top `top` share of the index (the relative High band)."""
    cut = index.quantile(1 - top)
    return set(index.index[index >= cut])


def overlap(a: set, b: set) -> float:
    """Share of a that is also in b."""
    return len(a & b) / len(a) if a else float("nan")


# ------------------------------------------------------------- validation

def behaviour(r: pd.DataFrame, step: int) -> dict:
    """Risk by hour / day type / point, and around congestion episodes."""
    by_hour = r.pivot_table(index="hour", columns="daytype", values="risk_index", aggfunc="mean")
    high_by_hour = r.assign(h=(r["band_relative"] == "High")).pivot_table(
        index="hour", columns="daytype", values="h", aggfunc="mean")
    by_point_hour = r[r["daytype"] == "weekday"].pivot_table(
        index="point_id", columns="hour", values="risk_index", aggfunc="mean")

    # Congestion episodes: consecutive readings (no gap) with speed_ratio < 0.85 lasting >= 30 min.
    phase = pd.Series("other", index=r.index)
    min_len = max(2, int(np.ceil(30 / step)))
    for _, g in r.sort_values("ts").groupby("point_id"):
        idx = g.index.to_numpy()
        cong = (g["speed_ratio"] < mf.LOW_MIN_RATIO).to_numpy()
        new_run = np.r_[True, (np.diff(g["ts"].to_numpy()).astype("timedelta64[m]").astype(int) > step) |
                        (cong[1:] != cong[:-1])]
        run_id = np.cumsum(new_run)
        ratio = g["speed_ratio"].to_numpy()
        for rid in np.unique(run_id[cong]):
            pos = np.where(run_id == rid)[0]
            if len(pos) < min_len:
                continue
            start, end = pos[0], pos[-1]
            seg_min = ratio[pos].min()
            phase.loc[idx[pos]] = "inside episode"
            phase.loc[idx[pos[ratio[pos] <= seg_min + 0.05]]] = "peak (within 0.05 of episode minimum)"
            phase.loc[idx[[p for p in (start - 1, start, start + 1) if 0 <= p < len(idx)]]] = "onset (+/- 1 reading)"
            phase.loc[idx[[p for p in (end - 1, end, end + 1) if 0 <= p < len(idx)]]] = "end (+/- 1 reading)"
    free = r["speed_ratio"] >= mf.LOW_MIN_RATIO
    phase[(phase == "other") & free & r["hour"].between(0, 4)] = "night free flow (00-05h)"
    phase[(phase == "other") & free] = "daytime free flow"
    r = r.assign(phase=phase)
    by_phase = r.groupby("phase").agg(
        n=("risk_index", "size"), mean_index=("risk_index", "mean"),
        share_high=("band_relative", lambda b: (b == "High").mean()))
    by_phase_point = r.pivot_table(index="point_id", columns="phase", values="risk_index", aggfunc="mean")
    return {
        "mean_index_by_hour": by_hour.round(2).to_dict(),
        "share_high_by_hour": high_by_hour.round(3).to_dict(),
        "weekday_mean_index_by_point_and_hour": by_point_hour.round(1).to_dict(orient="index"),
        "by_phase": by_phase.round(3).to_dict(orient="index"),
        "by_phase_and_point": by_phase_point.round(1).to_dict(orient="index"),
        "phase_rules": {
            "episode": "consecutive readings (no collection gap) with speed_ratio < 0.85 lasting >= 30 min",
            "onset": "first episode reading and one reading either side",
            "end": "last episode reading and one reading either side",
            "peak": "episode readings within 0.05 of the episode's minimum speed_ratio",
        },
    }, r["phase"]


def congestion_overlap(r: pd.DataFrame) -> dict:
    out = {}
    for band in ("band_relative", "band_absolute"):
        ct = pd.crosstab(r[band], r["congestion_level"]).reindex(index=BANDS, columns=mf.LABELS).fillna(0)
        out[band] = {
            "counts": ct.astype(int).to_dict(orient="index"),
            "row_percent": (100 * ct.div(ct.sum(axis=1), axis=0)).round(1).to_dict(orient="index"),
        }
    valid = r.dropna(subset=["risk_index"])
    out["correlation_index_vs_speed_ratio"] = {
        "pearson": valid["risk_index"].corr(valid["speed_ratio"]),
        "spearman": valid["risk_index"].corr(valid["speed_ratio"], method="spearman"),
    }
    out["correlation_component_vs_speed_ratio_spearman"] = {
        c: r[c].corr(r["speed_ratio"], method="spearman") for c in ms.COMPONENTS}
    high = r["band_relative"] == "High"
    out["high_readings_by_congestion_level_percent"] = (
        100 * r.loc[high, "congestion_level"].value_counts(normalize=True)).round(1).to_dict()
    out["high_readings_in_free_flow"] = int((high & (r["speed_ratio"] >= mf.LOW_MIN_RATIO)).sum())
    return out


def sensitivity(readings: pd.DataFrame, base: pd.DataFrame, cfg: ms.SafetyConfig) -> dict:
    """Overlap of the top-10% set when each weight is changed by +/-50%, and with a 2-hour window."""
    base_high = high_set(base["risk_index"])
    scores = base[[f"{c}_score" for c in ms.COMPONENTS]].rename(columns=lambda c: c[:-6])
    rows = []
    for c in ms.COMPONENTS:
        for factor in (0.5, 1.5):
            w = {**cfg.weights, c: cfg.weights[c] * factor}
            idx = ms.risk_index(scores, w, cfg.min_components)
            rows.append({"variant": f"{c} weight x{factor}", "high_overlap": overlap(base_high, high_set(idx))})
    alt = ms.SafetyConfig(step_min=cfg.step_min, window_min=2 * cfg.window_min, weights=cfg.weights)
    comp2 = ms.compute_components(readings, alt)
    idx2 = ms.risk_index(ms.percentile_scores(comp2), alt.weights, alt.min_components)
    rows.append({"variant": f"window {2 * cfg.window_min} min (vs {cfg.window_min})",
                 "high_overlap": overlap(base_high, high_set(idx2))})
    return {"baseline": f"top 10% of the index, window {cfg.window_min} min, equal weights",
            "overlap_definition": "share of the baseline High readings that are still High in the variant",
            "variants": rows}


def isolation_forest(r: pd.DataFrame) -> dict:
    valid = r.dropna(subset=["risk_index"])
    out = {}
    for name, x in (("percentile_scores", valid[[f"{c}_score" for c in ms.COMPONENTS]].fillna(0)),
                    ("raw_components_standardised",
                     ((valid[ms.COMPONENTS] - valid[ms.COMPONENTS].mean()) / valid[ms.COMPONENTS].std()).fillna(0))):
        forest = IsolationForest(n_estimators=300, contamination=0.10, random_state=42).fit(x)
        anomaly = pd.Series(-forest.score_samples(x), index=valid.index)
        out[name] = {"top10_overlap_with_index_high": overlap(high_set(valid["risk_index"]), high_set(anomaly))}
    out["note"] = "Isolation Forest on the same four components; overlap = share of our top-10% that it also ranks top-10%."
    return out


def incident_proxy(r: pd.DataFrame, readings: pd.DataFrame, incidents_csv: Path, radius_km: float) -> dict:
    """WEAK proxy: is the index higher when a TomTom incident is active nearby? Not accident validation."""
    variants = {
        "any incident (jam, closure, road works)": dict(),
        "short-lived jams only (observed <= 3 h)": dict(categories=["Jam"], max_hours=3),
        "closures and road works only": dict(categories=["Road Closed", "Road Works"]),
    }
    out = {"warning": "Weak evidence only: TomTom incidents are mostly jams, i.e. congestion itself. "
                      "This is not validation against accidents.",
           "radius_km": radius_km}
    for label, kw in variants.items():
        ctx = ms.incident_context(r, readings, str(incidents_csv), radius_km, **kw)
        covered = ctx["incident_count"].notna() & r["risk_index"].notna()
        near = covered & (ctx["incident_count"] > 0)
        far = covered & (ctx["incident_count"] == 0)
        high = r["band_relative"] == "High"
        # Same comparison within each congestion level, so congestion does not explain the gap.
        by_level = {}
        for lvl in mf.LABELS:
            m = r["congestion_level"] == lvl
            by_level[lvl] = {
                "mean_index_near": r.loc[near & m, "risk_index"].mean(),
                "mean_index_far": r.loc[far & m, "risk_index"].mean(),
                "n_near": int((near & m).sum()), "n_far": int((far & m).sum()),
            }
        out[label] = {
            "readings_covered": int(covered.sum()),
            "readings_with_incident": int(near.sum()),
            "share_with_incident": near.sum() / covered.sum(),
            "mean_index_near": r.loc[near, "risk_index"].mean(),
            "mean_index_far": r.loc[far, "risk_index"].mean(),
            "share_high_near": high[near].mean(),
            "share_high_far": high[far].mean(),
            "within_congestion_level": by_level,
        }
    return out


def top_moments(r: pd.DataFrame, n: int = 10) -> list[dict]:
    """Highest index, at most one per point within 2 hours, with speeds before / at / after."""
    r = r.sort_values(["point_id", "ts"])
    r = r.assign(speed_kmh_next=r.groupby("point_id")["speed_kmh"].shift(-1),
                 ts_next=r.groupby("point_id")["ts"].shift(-1))
    picked = []
    for _, row in r.dropna(subset=["risk_index"]).sort_values("risk_index", ascending=False).iterrows():
        if any(p["point_id"] == row["point_id"] and abs(p["ts"] - row["ts"]) < pd.Timedelta(hours=2) for p in picked):
            continue
        picked.append(row)
        if len(picked) == n:
            break
    return [{
        "point_id": p["point_id"], "time": str(p["ts"]), "reading_time": str(p["reading_ts"]),
        "risk_index": p["risk_index"], "main_reason": p["main_reason"],
        "speed_before_kmh": p["speed_kmh_prev"], "speed_kmh": p["speed_kmh"], "speed_after_kmh": p["speed_kmh_next"],
        "speed_ratio": p["speed_ratio"], "usual_ratio": p["usual_ratio"], "congestion_level": p["congestion_level"],
        "variability": p["variability"], "sudden_drop": p["sudden_drop"],
        "neighbour": p["neighbour"], "unusual": p["unusual"],
        "band_absolute": p["band_absolute"],
    } for p in picked]


def compare_steps(r10: pd.DataFrame, r30: pd.DataFrame) -> dict:
    """What the 30-min grid loses compared with the raw 10-min data."""
    h10 = r10[r10["band_relative"] == "High"]
    h30 = r30[r30["band_relative"] == "High"]
    # A 10-min High moment is "kept" if its 30-min slot (nearest grid time) is High for that point.
    slot = h10["reading_ts"].dt.round("30min")
    keys30 = set(zip(h30["point_id"], h30["ts"]))
    kept = np.mean([(p, t) in keys30 for p, t in zip(h10["point_id"], slot)])
    return {
        "readings": {"10min": int(len(r10)), "30min": int(len(r30))},
        "component_p95": {
            "10min": r10[ms.COMPONENTS].quantile(0.95).to_dict(),
            "30min": r30[ms.COMPONENTS].quantile(0.95).to_dict(),
        },
        "largest_sudden_drop": {"10min": r10["sudden_drop"].max(), "30min": r30["sudden_drop"].max()},
        "share_10min_high_moments_still_high_at_30min": kept,
        "main_reason_of_high_percent": {
            "10min": (100 * h10["main_reason"].value_counts(normalize=True)).round(1).to_dict(),
            "30min": (100 * h30["main_reason"].value_counts(normalize=True)).round(1).to_dict(),
        },
        "correlation_index_vs_speed_ratio_spearman": {
            "10min": r10["risk_index"].corr(r10["speed_ratio"], method="spearman"),
            "30min": r30["risk_index"].corr(r30["speed_ratio"], method="spearman"),
        },
        "note": "At 30 min the variability window (1 h) holds only 2 readings, and a 'sudden drop' is measured "
                "over 30 min, so short shock-like changes are averaged away or missed.",
    }


# ------------------------------------------------------------------ charts

def charts(r: pd.DataFrame, sens: dict, iso: dict, steps: dict, out: Path):
    # Risk by hour, weekday vs weekend, with the mean speed ratio for reference.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for dt, color in (("weekday", "#0d5a4e"), ("weekend", "#d6a13b")):
        g = r[r["daytype"] == dt].groupby("hour")
        ax.plot(g["risk_index"].mean(), color=color, lw=2, label=f"risk index, {dt}")
        ax.plot(100 * g["speed_ratio"].mean(), color=color, lw=1, ls=":", label=f"speed ratio x100, {dt}")
    ax.set_xticks(range(24))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean")
    ax.set_title("Safety risk indicator by hour (mean over all points) - not accident prediction")
    ax.legend(fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "safety_risk_by_hour.png", dpi=150)
    plt.close(fig)

    # Risk by point: mean index and the main reason of its High readings.
    high = r[r["band_relative"] == "High"]
    mix = pd.crosstab(high["point_id"], high["main_reason"], normalize="index").mul(100)
    share = (r.groupby("point_id")["band_relative"].apply(lambda b: (b == "High").mean()) * 100)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    r.groupby("point_id")["risk_index"].mean().plot.barh(ax=axes[0], color="#0d5a4e")
    for i, v in enumerate(share.reindex(r.groupby("point_id")["risk_index"].mean().index)):
        axes[0].text(1, i, f"{v:.0f}% High", va="center", fontsize=7, color="white")
    axes[0].set_title("Mean risk index (label: share of High readings)", fontsize=10)
    axes[0].set_ylabel("")
    mix.plot.barh(stacked=True, ax=axes[1], colormap="tab10", width=0.8)
    axes[1].set_title("Main reason of the High readings (%)", fontsize=10)
    axes[1].set_ylabel("")
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    axes[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(out / "safety_risk_by_point.png", dpi=150)
    plt.close(fig)

    # Component distributions (non-zero values; share of zeros in the title).
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, c in zip(axes, ms.COMPONENTS):
        x = r[c].dropna()
        ax.hist(x[x > 0], bins=40, color="#0d5a4e")
        ax.set_yscale("log")
        ax.set_title(f"{c}\n({(x == 0).mean():.0%} of readings are 0)", fontsize=9)
    fig.suptitle("Raw component distributions (non-zero values, log scale)")
    fig.tight_layout()
    fig.savefig(out / "safety_component_distributions.png", dpi=150)
    plt.close(fig)

    # Sensitivity summary.
    labels = [v["variant"] for v in sens["variants"]] + [
        "Isolation Forest (percentile scores)", "Isolation Forest (raw components)", "30-min grid"]
    values = [v["high_overlap"] for v in sens["variants"]] + [
        iso["percentile_scores"]["top10_overlap_with_index_high"],
        iso["raw_components_standardised"]["top10_overlap_with_index_high"],
        steps["share_10min_high_moments_still_high_at_30min"]]
    colors = ["#0d5a4e"] * len(sens["variants"]) + ["#8a96a3", "#8a96a3", "#d6a13b"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels[::-1], [100 * v for v in values[::-1]], color=colors[::-1])
    for i, v in enumerate(values[::-1]):
        ax.text(100 * v + 1, i, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of the baseline top-10% (High) readings that are also top-10%")
    ax.set_title("Robustness of the High set")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "safety_sensitivity.png", dpi=150)
    plt.close(fig)


def chart_timeline(r: pd.DataFrame, out: Path):
    """One weekday at the point/day with the deepest congestion episode."""
    wk = r[r["daytype"] == "weekday"].copy()
    wk["date"] = wk["ts"].dt.normalize()
    day_min = wk.groupby(["point_id", "date"])["speed_ratio"].min()
    counts = wk.groupby(["point_id", "date"]).size()
    point, date = day_min[counts >= 0.9 * counts.max()].idxmin()
    g = r[(r["point_id"] == point) & (r["ts"] >= date + pd.Timedelta(hours=5)) & (r["ts"] < date + pd.Timedelta(hours=23))]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.plot(g["ts"], g["speed_kmh"], color="#4565ef", lw=1.8, label="speed (km/h)")
    ax.set_ylabel("speed (km/h)", color="#4565ef")
    ax2 = ax.twinx()
    ax2.fill_between(g["ts"], g["risk_index"], color="#d64f49", alpha=0.25, step="mid", label="risk index")
    hi = g[g["band_relative"] == "High"]
    ax2.scatter(hi["ts"], hi["risk_index"], color="#d64f49", s=14, zorder=3, label="High (top 10%)")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("safety risk index (0-100)", color="#d64f49")
    ax.set_title(f"{point.replace('_', ' ')} on {date:%a %d %b}: speed and safety risk indicator")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "safety_timeline_example.png", dpi=150)
    plt.close(fig)
    return {"point_id": point, "date": str(date.date())}


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="Raw snapshot CSV (10-min data).")
    ap.add_argument("--incidents", default=str(DEFAULT_INCIDENTS), help="jeddah_incidents.csv (read only).")
    ap.add_argument("--step", type=int, default=10, help="Grid step of the data (default 10 = raw readings).")
    ap.add_argument("--window", type=int, default=60, help="Variability window in minutes (default 60).")
    ap.add_argument("--with-incidents", action="store_true",
                    help="Also compute an index that includes the optional incident component.")
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()

    out, charts_dir = Path(args.out) / "outputs", Path(args.out) / "charts"
    out.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    incidents = Path(args.incidents) if Path(args.incidents).exists() else None

    cfg = ms.SafetyConfig(step_min=args.step, window_min=args.window, use_incidents=args.with_incidents)
    readings, info = load(Path(args.csv), args.step)
    r = ms.build(readings, cfg, str(incidents) if incidents else None)
    print(f"{len(r):,} readings at step {args.step} min; index for {r['risk_index'].notna().sum():,}")

    order = ms.road_order(readings)
    pts = readings.groupby("point_id")[["lat", "lon"]].first()
    spacing = {road: [{"from": a, "to": b, "km": ms.haversine_km(*pts.loc[a], *pts.loc[b])}
                      for a, b in zip(p, p[1:])] for road, p in order.items()}

    beh, phase = behaviour(r, args.step)
    over = congestion_overlap(r)
    sens = sensitivity(readings, r, cfg)
    iso = isolation_forest(r)
    proxy = incident_proxy(r, readings, incidents, cfg.incident_radius_km) if incidents else {"skipped": "no incidents file"}
    top = top_moments(r)

    # 30-min grid comparison (same code, step 30).
    cfg30 = ms.SafetyConfig(step_min=30, window_min=args.window)
    readings30, _ = load(Path(args.csv), 30)
    r30 = ms.build(readings30, cfg30)
    steps = compare_steps(r, r30)
    mf.set_step(args.step)

    charts(r, sens, iso, steps, charts_dir)
    example = chart_timeline(r, charts_dir)

    incident_component = None
    if incidents:
        ic = r["incident_count"]
        incident_component = {
            "in_main_index": False,
            "readings_covered": int(ic.notna().sum()),
            "readings_affected": int((ic > 0).sum()),
            "note": "Optional component (flag --with-incidents). Incidents data starts 10 Oct 00:40 local; "
                    "earlier readings are unknown.",
        }

    r.assign(phase=phase).drop(columns=["daytype"]).to_csv(out / "safety_risk.csv", index=False)
    validation = clean({
        "what_this_is": "Safety risk indicator from speed patterns linked to crash risk in the literature "
                        "(Choudhary et al. 2018; Parsa et al. 2020). Not accident prediction; "
                        "not validated against crash data.",
        "data": {"source": info["source"], "step_min": args.step, "readings": len(r),
                 "first": info["first_reading"], "last": info["last_reading"]},
        "config": {"window_min": cfg.window_min, "weights": cfg.weights, "min_components": cfg.min_components,
                   "relative_bands": {"medium_from_percentile": cfg.relative_bands[0],
                                      "high_from_percentile": cfg.relative_bands[1]},
                   "absolute_cutoffs_medium_high": cfg.absolute_cutoffs,
                   "incident_radius_km": cfg.incident_radius_km},
        "road_order_north_to_south": order,
        "neighbour_spacing_km": spacing,
        "component_summary": r[ms.COMPONENTS].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "share_of_zero": (r[ms.COMPONENTS] == 0).mean().to_dict(),
        "bands": {"relative": r["band_relative"].value_counts().to_dict(),
                  "absolute": r["band_absolute"].value_counts().to_dict(),
                  "relative_vs_absolute": pd.crosstab(r["band_relative"], r["band_absolute"]).to_dict(orient="index")},
        "main_reason_all": r["main_reason"].value_counts().to_dict(),
        "main_reason_high": r.loc[r["band_relative"] == "High", "main_reason"].value_counts().to_dict(),
        "a_behaviour": beh,
        "b_congestion_overlap": over,
        "c_sensitivity": sens,
        "d_isolation_forest": iso,
        "e_incident_proxy": proxy,
        "e_incident_component": incident_component,
        "f_top10": top,
        "g_30min_vs_10min": steps,
        "timeline_example": example,
    })
    (out / "safety_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote outputs/safety_risk.csv, outputs/safety_validation.json and charts/safety_*.png")


if __name__ == "__main__":
    main()
