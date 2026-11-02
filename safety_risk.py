"""
Build and check the Masar SAFETY RISK INDICATOR offline (not accident prediction).

    python safety_risk.py                                   # raw 10-min snapshot, defaults
    python safety_risk.py --csv <snapshot> --step 10 --window 60

Writes:
    outputs/safety_risk.csv          per reading: components, scores, raw_index (phase 1),
                                     risk_index (main, level-adjusted), bands, main reason
    outputs/safety_usual.csv         usual risk per point x day type x hour (served by the web app)
    outputs/safety_validation.json   behaviour, overlap with congestion, sensitivity,
                                     Isolation Forest cross-check, incident proxy, top moments,
                                     10-min vs 30-min comparison, raw vs adjusted
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
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else round(float(value), 4)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def load(csv: Path, step: int):
    mf.set_step(step)
    return mf.load_data(str(csv))


def top_set(index: pd.Series, top: float = 0.10) -> set:
    """Readings in the top `top` share of an index."""
    return set(index.index[index >= index.quantile(1 - top)])


def high_set(index: pd.Series, cfg: ms.SafetyConfig) -> set:
    """Readings in the High band (fixed cut-off)."""
    return set(index.index[index >= cfg.band_cutoffs[1]])


def overlap(a: set, b: set) -> float:
    """Share of a that is also in b."""
    return len(a & b) / len(a) if a else float("nan")


def in_congestion(r: pd.DataFrame, rows) -> float:
    return float((r.loc[list(rows), "speed_ratio"] < mf.LOW_MIN_RATIO).mean()) if len(rows) else float("nan")


# ------------------------------------------------------------- validation

def behaviour(r: pd.DataFrame, step: int) -> tuple[dict, pd.Series]:
    """Risk by hour / day type / point, and around congestion episodes."""
    is_high = r["band"] == "High"
    by_hour = r.pivot_table(index="hour", columns="daytype", values="risk_index", aggfunc="mean")
    high_by_hour = r.assign(h=is_high).pivot_table(index="hour", columns="daytype", values="h", aggfunc="mean")
    by_point_hour = r[r["daytype"] == "weekday"].pivot_table(
        index="point_id", columns="hour", values="risk_index", aggfunc="mean")

    # Congestion episodes: consecutive readings (no gap) with speed_ratio < 0.85 lasting >= 30 min.
    phase = pd.Series("other", index=r.index)
    min_len = max(2, int(np.ceil(30 / step)))
    for _, g in r.sort_values("ts").groupby("point_id"):
        idx = g.index.to_numpy()
        cong = (g["speed_ratio"] < mf.LOW_MIN_RATIO).to_numpy()
        gaps = np.diff(g["ts"].to_numpy()).astype("timedelta64[m]").astype(int) > step
        run_id = np.cumsum(np.r_[True, gaps | (cong[1:] != cong[:-1])])
        ratio = g["speed_ratio"].to_numpy()
        for rid in np.unique(run_id[cong]):
            pos = np.where(run_id == rid)[0]
            if len(pos) < min_len:
                continue
            start, end = pos[0], pos[-1]
            phase.loc[idx[pos]] = "inside episode"
            phase.loc[idx[pos[ratio[pos] <= ratio[pos].min() + 0.05]]] = "peak (within 0.05 of episode minimum)"
            phase.loc[idx[[p for p in (start - 1, start, start + 1) if 0 <= p < len(idx)]]] = "onset (+/- 1 reading)"
            phase.loc[idx[[p for p in (end - 1, end, end + 1) if 0 <= p < len(idx)]]] = "end (+/- 1 reading)"
    free = r["speed_ratio"] >= mf.LOW_MIN_RATIO
    phase[(phase == "other") & free & r["hour"].between(0, 4)] = "night free flow (00-05h)"
    phase[(phase == "other") & free] = "daytime free flow"
    phase[phase == "other"] = "short dip (< 30 min)"
    r = r.assign(phase=phase, is_high=is_high)
    by_phase = r.groupby("phase").agg(n=("risk_index", "size"), mean_index=("risk_index", "mean"),
                                      share_high=("is_high", "mean"))
    return {
        "mean_index_by_hour": by_hour.round(2).to_dict(),
        "share_high_by_hour": high_by_hour.round(3).to_dict(),
        "weekday_mean_index_by_point_and_hour": by_point_hour.round(1).to_dict(orient="index"),
        "by_phase": by_phase.round(3).to_dict(orient="index"),
        "by_phase_and_point": r.pivot_table(index="point_id", columns="phase", values="risk_index",
                                            aggfunc="mean").round(1).to_dict(orient="index"),
        "phase_rules": {
            "episode": "consecutive readings (no collection gap) with speed_ratio < 0.85 lasting >= 30 min",
            "onset": "first episode reading and one reading either side",
            "end": "last episode reading and one reading either side",
            "peak": "episode readings within 0.05 of the episode's minimum speed_ratio",
            "short dip": "speed_ratio < 0.85 for less than 30 min",
        },
    }, phase


def congestion_overlap(r: pd.DataFrame, cfg: ms.SafetyConfig) -> dict:
    out = {}
    ct = pd.crosstab(r["band"], r["congestion_level"]).reindex(index=BANDS, columns=mf.LABELS).fillna(0)
    out["band_vs_congestion"] = {
        "counts": ct.astype(int).to_dict(orient="index"),
        "row_percent": (100 * ct.div(ct.sum(axis=1), axis=0)).round(1).to_dict(orient="index"),
    }
    valid = r.dropna(subset=["risk_index"])
    out["correlation_index_vs_speed_ratio"] = {
        "spearman": valid["risk_index"].corr(valid["speed_ratio"], method="spearman"),
        "pearson": valid["risk_index"].corr(valid["speed_ratio"]),
    }
    out["correlation_raw_index_vs_speed_ratio"] = {
        "spearman": valid["raw_index"].corr(valid["speed_ratio"], method="spearman"),
        "pearson": valid["raw_index"].corr(valid["speed_ratio"]),
    }
    level = ms.speed_level(r["speed_ratio"], cfg.level_bins)
    out["component_excess_vs_speed_ratio_spearman"] = {
        c: (r[f"{c}_score"] - r[f"{c}_score"].groupby(level, observed=True).transform("mean")).clip(lower=0)
        .corr(r["speed_ratio"], method="spearman") for c in ms.COMPONENTS}
    high = r["band"] == "High"
    out["high_by_congestion_level_percent"] = (100 * r.loc[high, "congestion_level"].value_counts(normalize=True)).round(1).to_dict()
    out["high_by_speed_level_percent"] = (100 * level[high].value_counts(normalize=True).sort_index()).round(1).astype(float).rename(str).to_dict()
    return out


def sensitivity(readings: pd.DataFrame, r: pd.DataFrame, cfg: ms.SafetyConfig) -> dict:
    """Overlap of the High set (fixed cut-off) and of the top 10% when each weight is changed by
    +/-50% and with a 2-hour window."""
    scores = r[[f"{c}_score" for c in ms.COMPONENTS]].rename(columns=lambda c: c[:-6])
    base_high, base_top = high_set(r["risk_index"], cfg), top_set(r["risk_index"])
    rows = []

    def variant(label, idx):
        rows.append({"variant": label, "high_overlap": overlap(base_high, high_set(idx, cfg)),
                     "top10_overlap": overlap(base_top, top_set(idx)),
                     "high_count": len(high_set(idx, cfg))})

    for c in ms.COMPONENTS:
        for factor in (0.5, 1.5):
            w = {**cfg.weights, c: cfg.weights[c] * factor}
            variant(f"{c} weight x{factor}",
                    ms.main_index(ms.weighted_score(scores, w, cfg.min_components), r["speed_ratio"], r["sudden_drop"], cfg)[0])
    alt = ms.SafetyConfig(step_min=cfg.step_min, window_min=2 * cfg.window_min)
    comp2 = ms.compute_components(readings, alt)
    s2 = ms.percentile_scores(comp2, by="point_id", positive_only=True)
    idx2 = ms.main_index(ms.weighted_score(s2, alt.weights, alt.min_components), comp2["speed_ratio"], comp2["sudden_drop"], alt)[0]
    variant(f"window {2 * cfg.window_min} min (vs {cfg.window_min})", idx2)

    # Same check for the phase-1 raw index, for the before/after table.
    gscores = ms.percentile_scores(r.assign(sudden_drop=r["sudden_drop_raw"]))
    raw_top = top_set(r["raw_index"])
    raw_rows = []
    for c in ms.COMPONENTS:
        for factor in (0.5, 1.5):
            w = {**cfg.raw_weights, c: factor}
            raw_rows.append(overlap(raw_top, top_set(ms.risk_index(gscores, w, cfg.min_components))))
    return {
        "baseline": f"High band (index >= {cfg.band_cutoffs[1]}), window {cfg.window_min} min, weights {cfg.weights}",
        "overlap_definition": "share of the baseline set that is still in the variant's set",
        "variants": rows,
        # Gate on the weights only: the 60-min window is part of the definition,
        # the 2-hour window is reported as a limitation.
        "min_high_overlap": min(v["high_overlap"] for v in rows if "weight" in v["variant"]),
        "min_top10_overlap": min(v["top10_overlap"] for v in rows if "weight" in v["variant"]),
        "window_120_high_overlap": next(v["high_overlap"] for v in rows if v["variant"].startswith("window")),
        "raw_index_min_top10_overlap_weights": min(raw_rows),
    }


def isolation_forest(r: pd.DataFrame, cfg: ms.SafetyConfig) -> dict:
    valid = r.dropna(subset=["risk_index"])
    x = valid[[f"{c}_score" for c in ms.COMPONENTS]].fillna(0)
    level = ms.speed_level(valid["speed_ratio"], cfg.level_bins)
    x_rel = x - x.groupby(level, observed=True).transform("mean")    # same level adjustment as the index
    out = {}
    for name, data in (("per_point_scores", x), ("per_point_scores_minus_level_mean", x_rel)):
        forest = IsolationForest(n_estimators=300, contamination=0.10, random_state=42).fit(data)
        anomaly = pd.Series(-forest.score_samples(data), index=valid.index)
        out[name] = {"top10_overlap_with_index_top10": overlap(top_set(valid["risk_index"]), top_set(anomaly)),
                     "high_band_in_forest_top10": overlap(high_set(valid["risk_index"], cfg), top_set(anomaly))}
    out["note"] = "Isolation Forest on the same four components (per-point scores)."
    return out


def incident_proxy(r: pd.DataFrame, readings: pd.DataFrame, incidents_csv: Path, radius_km: float) -> dict:
    """WEAK proxy: is the index higher when a TomTom incident is active nearby? Not accident validation."""
    variants = {
        "any incident (jam, closure, road works)": dict(),
        "short-lived jams only (observed <= 3 h)": dict(categories=["Jam"], max_hours=3),
        "closures and road works only": dict(categories=["Road Closed", "Road Works"]),
    }
    out = {"warning": "Weak evidence only: TomTom incidents are mostly jams, i.e. congestion itself, and the "
                      "data has no accident incidents. This is not validation against accidents.",
           "radius_km": radius_km}
    high = r["band"] == "High"
    for label, kw in variants.items():
        ctx = ms.incident_context(r, readings, str(incidents_csv), radius_km, **kw)
        covered = ctx["incident_count"].notna() & r["risk_index"].notna()
        near = covered & (ctx["incident_count"] > 0)
        far = covered & (ctx["incident_count"] == 0)
        out[label] = {
            "readings_covered": int(covered.sum()),
            "readings_with_incident": int(near.sum()),
            "share_with_incident": near.sum() / covered.sum(),
            "mean_index_near": r.loc[near, "risk_index"].mean(),
            "mean_index_far": r.loc[far, "risk_index"].mean(),
            "share_high_near": high[near].mean(),
            "share_high_far": high[far].mean(),
            "within_congestion_level": {lvl: {
                "mean_index_near": r.loc[near & (r["congestion_level"] == lvl), "risk_index"].mean(),
                "mean_index_far": r.loc[far & (r["congestion_level"] == lvl), "risk_index"].mean(),
                "n_near": int((near & (r["congestion_level"] == lvl)).sum()),
                "n_far": int((far & (r["congestion_level"] == lvl)).sum()),
            } for lvl in mf.LABELS},
        }
    return out


def top_moments(r: pd.DataFrame, n: int = 10, column: str = "risk_index") -> list[dict]:
    """Highest index, at most one per point within 2 hours, with speeds before / at / after."""
    r = r.sort_values(["point_id", "ts"])
    r = r.assign(speed_kmh_next=r.groupby("point_id")["speed_kmh"].shift(-1))
    picked = []
    for _, row in r.dropna(subset=[column]).sort_values(column, ascending=False).iterrows():
        if any(p["point_id"] == row["point_id"] and abs(p["ts"] - row["ts"]) < pd.Timedelta(hours=2) for p in picked):
            continue
        picked.append(row)
        if len(picked) == n:
            break
    return [{
        "point_id": p["point_id"], "time": str(p["ts"]), "reading_time": str(p["reading_ts"]),
        "risk_index": p["risk_index"], "raw_index": p["raw_index"], "band": p["band"], "main_reason": p["main_reason"],
        "speed_before_kmh": p["speed_kmh_prev"], "speed_kmh": p["speed_kmh"], "speed_after_kmh": p["speed_kmh_next"],
        "speed_ratio": p["speed_ratio"], "usual_ratio": p["usual_ratio"], "congestion_level": p["congestion_level"],
        "variability": p["variability"], "sudden_drop": p["sudden_drop"], "neighbour": p["neighbour"],
        "unusual": p["unusual"],
    } for p in picked]


def compare_steps(r10: pd.DataFrame, r30: pd.DataFrame) -> dict:
    """What the 30-min grid loses compared with the raw 10-min data."""
    h10 = r10[r10["band"] == "High"]
    h30 = r30[r30["band"] == "High"]
    slot = h10["reading_ts"].dt.round("30min")
    keys30 = set(zip(h30["point_id"], h30["ts"]))
    return {
        "readings": {"10min": int(len(r10)), "30min": int(len(r30))},
        "share_high": {"10min": float((r10["band"] == "High").mean()), "30min": float((r30["band"] == "High").mean())},
        "component_p95": {"10min": r10[ms.COMPONENTS].quantile(0.95).to_dict(),
                          "30min": r30[ms.COMPONENTS].quantile(0.95).to_dict()},
        "largest_sudden_drop": {"10min": r10["sudden_drop"].max(), "30min": r30["sudden_drop"].max()},
        "share_10min_high_moments_still_high_at_30min": float(np.mean([(p, t) in keys30 for p, t in zip(h10["point_id"], slot)])) if len(h10) else None,
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


def usual_table(r: pd.DataFrame, cfg: ms.SafetyConfig) -> pd.DataFrame:
    """Usual risk per point x day type x hour - what the web app shows for a chosen date and hour.

    Risk moments are rare, short events, so the MEAN index of an hour is always
    low (max ~20, never Medium). The app therefore shows the 90th percentile
    ("on the worse days at this hour the index reaches at least this"), banded
    with the same fixed cut-offs, plus how often the hour was Medium or High.
    The reason is the most frequent main reason among the elevated readings.
    """
    rows = []
    for (pid, dt, hour), g in r.dropna(subset=["risk_index"]).groupby(["point_id", "daytype", "hour"]):
        elevated = g.loc[g["band"].isin(["Medium", "High"]), "main_reason"]
        nonzero = g.loc[g["main_reason"].isin(ms.REASONS.values()), "main_reason"]
        reasons = elevated if not elevated.empty else nonzero
        p90 = float(g["risk_index"].quantile(0.90))
        rows.append({
            "point_id": pid, "daytype": dt, "hour": hour,
            "readings": len(g), "days": g["ts"].dt.normalize().nunique(),
            "p90_index": p90,
            "band": ms.fixed_bands(pd.Series([p90]), cfg.band_cutoffs).iloc[0],
            "mean_index": g["risk_index"].mean(),
            "max_index": g["risk_index"].max(),
            "share_medium_or_high": (g["band"] != "Low").mean(),
            "share_high": (g["band"] == "High").mean(),
            "main_reason": reasons.mode().iloc[0] if not reasons.empty else "none",
            "mean_speed_ratio": g["speed_ratio"].mean(),
        })
    return pd.DataFrame(rows)


def after_gap_window(r: pd.DataFrame, minutes: int) -> pd.Series:
    """True for readings within `minutes` after the end of a collection gap at that point."""
    gap_end = r["ts"].where(r["after_gap"])
    last_gap_end = gap_end.groupby(r["point_id"]).ffill()
    return last_gap_end.notna() & ((r["ts"] - last_gap_end) <= pd.Timedelta(minutes=minutes))


def variant_comparison(r: pd.DataFrame, cfg: ms.SafetyConfig) -> list[dict]:
    """The four level-adjustment variants discussed before choosing option 1, on the same scores."""
    scores = r[[f"{c}_score" for c in ms.COMPONENTS]].rename(columns=lambda c: c[:-6])
    nine = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.95, 0.9999, 1.01]
    cong = r["speed_ratio"] < mf.LOW_MIN_RATIO
    full = r["speed_ratio"] >= 0.9999
    out = []
    for label, bins, recovery in (("9 speed bands", nine, False),
                                  ("3 congestion levels", list(ms.LEVEL_BINS), False),
                                  ("9 speed bands + recovery rule", nine, True),
                                  ("3 congestion levels + recovery rule (chosen)", list(ms.LEVEL_BINS), True)):
        vcfg = ms.SafetyConfig(level_bins=bins, recovery_rule=recovery)

        def idx_for(w):
            return ms.main_index(ms.weighted_score(scores, w, vcfg.min_components),
                                 r["speed_ratio"], r["sudden_drop"], vcfg)[0]

        base = idx_for(vcfg.weights)
        hi, top = high_set(base, vcfg), top_set(base)
        ov, ovt = [], []
        for c in ms.COMPONENTS:
            for f in (0.5, 1.5):
                v = idx_for({**vcfg.weights, c: vcfg.weights[c] * f})
                ov.append(overlap(hi, high_set(v, vcfg)))
                ovt.append(overlap(top, top_set(v)))
        hl = list(hi)
        out.append({
            "variant": label,
            "spearman_vs_speed_ratio": base.corr(r["speed_ratio"], method="spearman"),
            "share_high": len(hi) / base.notna().sum(),
            "high_at_full_free_flow": float(full[hl].mean()) if hl else 0.0,
            "high_in_congestion": float(cong[hl].mean()) if hl else 0.0,
            "night_free_flow_share_high": float((base[(r["hour"] < 5) & ~cong] >= vcfg.band_cutoffs[1]).mean()),
            "worst_high_overlap_weights": min(ov),
            "worst_top10_overlap_weights": min(ovt),
        })
    return out


def before_after(r: pd.DataFrame, sens: dict, cfg: ms.SafetyConfig) -> dict:
    raw_high = r["raw_band_relative"] == "High"
    high = r["band"] == "High"
    top = r["risk_index"] >= r["risk_index"].quantile(0.9)
    cong = r["speed_ratio"] < mf.LOW_MIN_RATIO
    return {
        "spearman_vs_speed_ratio": {"raw_index": r["raw_index"].corr(r["speed_ratio"], method="spearman"),
                                    "risk_index": r["risk_index"].corr(r["speed_ratio"], method="spearman")},
        "share_high_in_congestion": {"raw_index (top 10%)": cong[raw_high].mean(),
                                     "risk_index (High band)": cong[high].mean(),
                                     "risk_index (top 10%)": cong[top].mean()},
        "share_high_overall": {"raw_index (top 10%)": raw_high.mean(), "risk_index (High band)": high.mean()},
        "sensitivity_min_overlap": {"raw_index (top 10%, weights)": sens["raw_index_min_top10_overlap_weights"],
                                    "risk_index (High band, weights + window)": sens["min_high_overlap"],
                                    "risk_index (top 10%, weights + window)": sens["min_top10_overlap"]},
        "high_share_per_point_percent": {
            "raw_index (top 10%)": (100 * raw_high.groupby(r["point_id"]).mean()).round(1).to_dict(),
            "risk_index (High band)": (100 * high.groupby(r["point_id"]).mean()).round(1).to_dict(),
        },
        "top10_moments_per_point": {
            "raw_index": pd.Series([m["point_id"] for m in top_moments(r, column="raw_index")]).value_counts().to_dict(),
            "risk_index": pd.Series([m["point_id"] for m in top_moments(r)]).value_counts().to_dict(),
        },
    }


# ------------------------------------------------------------------ charts

def charts(r: pd.DataFrame, sens: dict, iso: dict, steps: dict, ba: dict, out: Path):
    # Risk by hour, weekday vs weekend, with the mean speed ratio for reference.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for dt, color in (("weekday", "#0d5a4e"), ("weekend", "#d6a13b")):
        g = r[r["daytype"] == dt].groupby("hour")
        ax.plot(g["risk_index"].mean(), color=color, lw=2, label=f"risk index, {dt}")
        ax.plot(g["raw_index"].mean(), color=color, lw=1, ls="--", label=f"raw index (phase 1), {dt}")
        ax.plot(100 * (1 - g["speed_ratio"].mean()), color=color, lw=1, ls=":", label=f"slowdown (1 - speed ratio) x100, {dt}")
    ax.set_xticks(range(24))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean")
    ax.set_title("Safety risk indicator by hour (mean over all points) - not accident prediction")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "safety_risk_by_hour.png", dpi=150)
    plt.close(fig)

    # Risk by point: High share before (raw) and after, and the main reason of the High readings.
    per_point = pd.DataFrame(ba["high_share_per_point_percent"])
    high = r[r["band"] == "High"]
    mix = pd.crosstab(high["point_id"], high["main_reason"], normalize="index").mul(100)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
    per_point.plot.barh(ax=axes[0], color=["#8a96a3", "#0d5a4e"], width=0.8)
    axes[0].set_title("Share of High readings per point (%)", fontsize=10)
    axes[0].set_ylabel("")
    axes[0].legend(fontsize=7, loc="lower right")
    mix.reindex(per_point.index).plot.barh(stacked=True, ax=axes[1], colormap="tab10", width=0.8)
    axes[1].set_title("Main reason of the High readings, risk index (%)", fontsize=10)
    axes[1].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
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
        "Isolation Forest (level-adjusted scores)", "30-min grid"]
    values = [v["top10_overlap"] for v in sens["variants"]] + [
        iso["per_point_scores_minus_level_mean"]["top10_overlap_with_index_top10"],
        steps["share_10min_high_moments_still_high_at_30min"] or 0]
    colors = ["#0d5a4e"] * len(sens["variants"]) + ["#8a96a3", "#d6a13b"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels[::-1], [100 * v for v in values[::-1]], color=colors[::-1])
    for i, v in enumerate(values[::-1]):
        ax.text(100 * v + 1, i, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 110)
    ax.axvline(80, color="#d64f49", ls="--", lw=1)
    ax.set_xlabel("% of the baseline top-10% readings that stay in the top 10% (gate: 80%, weights only)")
    ax.set_title("Robustness of the ranking (risk index); the 2-hour window is not gated")
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
    ax2.step(g["ts"], g["raw_index"], where="mid", color="#8a96a3", lw=1, ls="--", label="raw index (phase 1)")
    ax2.fill_between(g["ts"], g["risk_index"], color="#d64f49", alpha=0.25, step="mid", label="risk index")
    hi = g[g["band"] == "High"]
    ax2.scatter(hi["ts"], hi["risk_index"], color="#d64f49", s=16, zorder=3, label="High band")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("safety risk index (0-100)", color="#d64f49")
    ax.set_title(f"{point.replace('_', ' ')} on {date:%a %d %b}: speed and safety risk indicator")
    lines, lines2 = ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()
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
    over = congestion_overlap(r, cfg)
    sens = sensitivity(readings, r, cfg)
    iso = isolation_forest(r, cfg)
    proxy = incident_proxy(r, readings, incidents, cfg.incident_radius_km) if incidents else {"skipped": "no incidents file"}
    top = top_moments(r)
    ba = before_after(r, sens, cfg)

    # 30-min grid comparison (same code, step 30).
    readings30, _ = load(Path(args.csv), 30)
    r30 = ms.build(readings30, ms.SafetyConfig(step_min=30, window_min=args.window))
    steps = compare_steps(r, r30)
    mf.set_step(args.step)

    charts(r, sens, iso, steps, ba, charts_dir)
    example = chart_timeline(r, charts_dir)

    incident_component = None
    if incidents:
        ic = r["incident_count"]
        incident_component = {
            "in_main_index": False, "readings_covered": int(ic.notna().sum()), "readings_affected": int((ic > 0).sum()),
            "note": "Optional component (flag --with-incidents). Incidents data starts 10 Oct 00:40 local; "
                    "earlier readings are unknown.",
        }

    variants = variant_comparison(r, cfg)
    high = r["band"] == "High"
    rho = over["correlation_index_vs_speed_ratio"]["spearman"]
    gate = {
        "spearman_vs_speed_ratio": rho,
        "abs_spearman_at_most_0.65": abs(rho) <= 0.65,
        "high_at_full_free_flow": int((high & (r["speed_ratio"] >= 0.9999)).sum()),
        "high_right_after_gap": int((high & r["after_gap"]).sum()),
        "high_within_one_hour_after_gap": int((high & after_gap_window(r, 60)).sum()),
        "min_top10_overlap_weights": sens["min_top10_overlap"],
        "min_top10_overlap_weights_at_least_0.80": sens["min_top10_overlap"] >= 0.80,
        "min_high_overlap_weights_not_gated": sens["min_high_overlap"],
        "high_count_range_weights": [min(v["high_count"] for v in sens["variants"] if "weight" in v["variant"]),
                                     max(v["high_count"] for v in sens["variants"] if "weight" in v["variant"])],
        "window_120_high_overlap_not_gated": sens["window_120_high_overlap"],
        "night_share_of_high": float((high & (r["hour"] < 5)).sum() / max(high.sum(), 1)),
        "night_share_of_high_below_0.15": bool((high & (r["hour"] < 5)).sum() / max(high.sum(), 1) < 0.15),
        "night_free_flow_mean_index": beh["by_phase"].get("night free flow (00-05h)", {}).get("mean_index"),
    }
    gate["passed"] = bool(gate["abs_spearman_at_most_0.65"] and gate["high_at_full_free_flow"] == 0
                          and gate["high_right_after_gap"] == 0 and gate["min_top10_overlap_weights_at_least_0.80"]
                          and gate["night_share_of_high_below_0.15"])

    usual = usual_table(r, cfg)
    usual.round(4).to_csv(out / "safety_usual.csv", index=False)
    r.assign(phase=phase).drop(columns=["daytype"]).to_csv(out / "safety_risk.csv", index=False)
    validation = clean({
        "what_this_is": "Safety risk indicator from speed patterns linked to crash risk in the literature "
                        "(Choudhary et al. 2018; Parsa et al. 2020): risk beyond what is usual for the traffic "
                        "level. Not accident prediction; not validated against crash data.",
        "data": {"source": info["source"], "step_min": args.step, "readings": len(r),
                 "first": info["first_reading"], "last": info["last_reading"]},
        "config": {"window_min": cfg.window_min, "weights": cfg.weights, "raw_weights": cfg.raw_weights,
                   "per_point_percentiles": cfg.per_point, "level_bins": cfg.level_bins,
                   "full_scale": cfg.full_scale, "band_cutoffs_medium_high": cfg.band_cutoffs,
                   "min_components": cfg.min_components, "incident_radius_km": cfg.incident_radius_km},
        "gate": gate,
        "before_after": ba,
        "variant_comparison": variants,
        "road_order_north_to_south": order,
        "neighbour_spacing_km": spacing,
        "component_summary": r[ms.COMPONENTS].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "index_summary": r["risk_index"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(),
        "share_index_zero": float((r["risk_index"] == 0).mean()),
        "bands": {"fixed": r["band"].value_counts().to_dict(), "relative_comparison": r["band_relative"].value_counts().to_dict()},
        "main_reason_all": r["main_reason"].value_counts().to_dict(),
        "main_reason_high": r.loc[r["band"] == "High", "main_reason"].value_counts().to_dict(),
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
    print(json.dumps(validation["gate"]))
    print("Wrote outputs/safety_risk.csv, outputs/safety_usual.csv, outputs/safety_validation.json, charts/safety_*.png")


if __name__ == "__main__":
    main()
