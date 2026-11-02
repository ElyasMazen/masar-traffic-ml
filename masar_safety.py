"""
Masar safety risk indicator - rule-based, from speed patterns.

This is a SAFETY RISK INDICATOR, not accident prediction: there is no
road-level crash data for these roads. It combines speed-pattern factors that
the literature links to higher crash risk:
  - Choudhary et al. (2018): speed variability is associated with crash risk.
  - Parsa et al. (2020): speed differences between upstream and downstream
    locations are among the strongest crash predictors (XGBoost + SHAP).

Components, per point and reading (all describe CHANGE or INSTABILITY, not the
congestion level itself):
  a) variability   coefficient of variation of speed over a rolling window
  b) sudden_drop   fall in speed_ratio since the previous reading (drops only)
  c) neighbour     |speed_ratio - speed_ratio of the adjacent point on the road| (approximate)
  d) unusual       how far speed_ratio is BELOW the usual ratio for point x hour x day type
  e) incident      (optional, off by default) TomTom incidents active within ~1 km

Each component is turned into a 0-1 percentile rank, combined with the weights
in WEIGHTS into a 0-100 Risk Index, and banded two ways (relative / absolute).

Reuses masar_features for loading, gap rules (lag_features) and the usual
tables. The grid step follows mf.STEP_MIN (10 for the raw data, 30 later).
Reads the data repo only; never calls the TomTom API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import masar_features as mf

# ------------------------------------------------------------------ config

COMPONENTS = ["variability", "sudden_drop", "neighbour", "unusual"]
REASONS = {
    "variability": "high variability",
    "sudden_drop": "sudden drop",
    "neighbour": "speed difference to neighbouring point",
    "unusual": "unusually slow for this hour",
}


@dataclass
class SafetyConfig:
    step_min: int = 10                 # grid step of the input data
    window_min: int = 60               # rolling window for the variability component
    weights: dict = field(default_factory=lambda: {c: 1.0 for c in COMPONENTS})   # equal weights
    min_components: int = 3            # an index needs at least this many components
    relative_bands: tuple = (0.70, 0.90)   # percentile of the index: >= 0.90 High, >= 0.70 Medium
    # Absolute cut-offs on the RAW components (see docs/SAFETY_INDICATOR.md for the reasoning).
    # High if any component reaches its High cut-off, or two reach their Medium cut-off.
    absolute_cutoffs: dict = field(default_factory=lambda: {
        #               Medium  High
        "variability": (0.10, 0.20),   # CV of speed within the window
        "sudden_drop": (0.15, 0.30),   # fall in speed_ratio since the previous reading
        "neighbour":   (0.30, 0.50),   # |ratio difference| to an adjacent point
        "unusual":     (0.15, 0.30),   # ratio below the usual ratio for this hour
    })
    incident_radius_km: float = 1.0
    use_incidents: bool = False        # optional component; never part of the main index by default


# Point order along each road (north -> south), worked out from the coordinates
# in road_order(). Travel direction is NOT known: TomTom's flow segment can be
# either carriageway and the CSV stores only the segment midpoint.

def road_order(readings: pd.DataFrame) -> dict[str, list[str]]:
    """Order the points of each road along its main axis (first principal component), north first."""
    pts = readings.groupby("point_id").agg(road=("road", "first"), lat=("lat", "first"), lon=("lon", "first"))
    order = {}
    for road, g in pts.groupby("road"):
        xy = np.c_[g["lon"] * np.cos(np.radians(g["lat"].mean())), g["lat"]]
        xy = xy - xy.mean(axis=0)
        axis = np.linalg.svd(xy, full_matrices=False)[2][0]
        proj = xy @ axis
        if np.corrcoef(proj, g["lat"])[0, 1] < 0:      # make "first" = northernmost
            proj = -proj
        order[road] = list(g.index[np.argsort(-proj)])
    return order


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


# -------------------------------------------------------------- components

def compute_components(readings: pd.DataFrame, cfg: SafetyConfig, usual: dict | None = None) -> pd.DataFrame:
    """Raw (un-normalised) components for every reading. readings come from mf.load_data."""
    d = readings.sort_values(["point_id", "ts"]).reset_index(drop=True)

    # b) sudden drop - uses mf.lag_features, so lag1 is NaN across a collection gap.
    lags = mf.lag_features(d)
    out = lags[["point_id", "road", "ts", "reading_ts", "speed_ratio", "congestion_level",
                "current_speed_kmh_t", "current_speed_kmh_lag1", "speed_ratio_lag1"]].copy()
    out = out.rename(columns={"current_speed_kmh_t": "speed_kmh", "current_speed_kmh_lag1": "speed_kmh_prev",
                              "speed_ratio_lag1": "speed_ratio_prev"})
    out["sudden_drop"] = (out["speed_ratio_prev"] - out["speed_ratio"]).clip(lower=0)   # NaN stays NaN

    # a) variability - CV of speed over a time-based window (t - window, t] per point.
    # Needs at least half the readings the window should hold (and >= 2).
    need = max(2, int(np.ceil(0.5 * cfg.window_min / cfg.step_min)))
    cv = []
    for _, g in d.groupby("point_id", sort=True):
        s = g.set_index("ts")["current_speed_kmh"].astype(float)
        roll = s.rolling(f"{cfg.window_min}min", min_periods=need)
        cv.append((roll.std(ddof=0) / roll.mean()).to_numpy())
    out["variability"] = np.concatenate(cv)

    # c) neighbour difference - max |ratio difference| to the adjacent point(s) on
    # the same road at the same grid time. Direction is unknown, so both sides count.
    order = road_order(d)
    ratio = d.pivot_table(index="ts", columns="point_id", values="speed_ratio")
    nb = {}
    for road, pts in order.items():
        for i, p in enumerate(pts):
            diffs = [(ratio[p] - ratio[q]).abs() for q in (pts[i - 1] if i > 0 else None,
                                                           pts[i + 1] if i < len(pts) - 1 else None) if q]
            nb[p] = pd.concat(diffs, axis=1).max(axis=1, skipna=True)
    nb_long = pd.DataFrame(nb).stack(future_stack=True).rename("neighbour").reset_index()
    nb_long.columns = ["ts", "point_id", "neighbour"]
    out = out.merge(nb_long, on=["ts", "point_id"], how="left")

    # d) unusual - shortfall below the usual ratio for point x hour x day type.
    usual = usual if usual is not None else mf.fit_usual(d)
    out["usual_ratio"] = mf.usual_ratio(usual, out["point_id"], out["ts"].dt.hour, mf.day_type(out["ts"].dt.dayofweek))
    out["unusual"] = (out["usual_ratio"] - out["speed_ratio"]).clip(lower=0)

    out["hour"] = out["ts"].dt.hour
    out["daytype"] = mf.day_type(out["ts"].dt.dayofweek)
    return out


def incident_context(comp: pd.DataFrame, readings: pd.DataFrame, incidents_csv: str,
                     radius_km: float = 1.0, categories=None, max_hours: float | None = None) -> pd.DataFrame:
    """e) Optional: number of TomTom incidents active within radius_km of the point at the reading time.

    An incident is 'active' between first_seen_utc and last_seen_utc (when the
    incidents collector actually observed it). Its location is the straight
    line between its start and end coordinates (the CSV has no full geometry).
    Readings before the incidents collection started are left NaN (unknown).

    categories / max_hours optionally keep only some incident types, or only
    incidents observed for at most max_hours (TomTom reports some "Jam"
    incidents as active for days, which says little about a given moment).
    """
    inc = pd.read_csv(incidents_csv, encoding="utf-8-sig")
    inc["first"] = pd.to_datetime(inc["first_seen_utc"])
    inc["last"] = pd.to_datetime(inc["last_seen_utc"])
    covered_from = inc["first"].min()
    if categories is not None:
        inc = inc[inc["category"].isin(categories)]
    if max_hours is not None:
        inc = inc[(inc["last"] - inc["first"]) <= pd.Timedelta(hours=max_hours)]
    pts = readings.groupby("point_id")[["lat", "lon"]].first()

    out = comp[["point_id", "reading_ts"]].copy()
    out["t_utc"] = out["reading_ts"] - pd.Timedelta(hours=3)   # timestamp_local is UTC+3
    out["incident_count"] = np.nan
    out["incident_categories"] = ""
    for pid, (plat, plon) in pts.iterrows():
        # distance from the point to the incident's start-end line (equirectangular, fine at km scale)
        kx = 111.32 * np.cos(np.radians(plat))
        ax, ay = (inc["start_lon"] - plon) * kx, (inc["start_lat"] - plat) * 110.57
        bx, by = (inc["end_lon"] - plon) * kx, (inc["end_lat"] - plat) * 110.57
        dx, dy = bx - ax, by - ay
        t = np.clip(-(ax * dx + ay * dy) / np.maximum(dx * dx + dy * dy, 1e-12), 0, 1)
        dist = np.hypot(ax + t * dx, ay + t * dy)
        near = inc[dist <= radius_km]
        rows = out.index[out["point_id"] == pid]
        tt = out.loc[rows, "t_utc"].to_numpy()[:, None]
        active = (tt >= near["first"].to_numpy()[None, :]) & (tt <= near["last"].to_numpy()[None, :])
        counts = active.sum(axis=1).astype(float)
        counts[out.loc[rows, "t_utc"].to_numpy() < covered_from] = np.nan
        out.loc[rows, "incident_count"] = counts
        cats = near["category"].to_numpy()
        out.loc[rows, "incident_categories"] = ["|".join(sorted(set(cats[a]))) for a in active]
    return out[["incident_count", "incident_categories"]]


# ----------------------------------------------------------------- the index

def percentile_scores(comp: pd.DataFrame, columns=COMPONENTS) -> pd.DataFrame:
    """Each component -> 0-1 percentile rank over all readings.

    Ties take their LOWEST rank, so a component that is 0 (no drop, not below
    usual) scores exactly 0 rather than a mid rank; the largest value scores 1.
    """
    scores = pd.DataFrame(index=comp.index)
    for c in columns:
        x = comp[c]
        r = x.rank(method="min")
        n = x.notna().sum()
        scores[c] = (r - 1) / max(n - 1, 1)
    return scores


def risk_index(scores: pd.DataFrame, weights: dict, min_components: int = 3) -> pd.Series:
    """Weighted mean of the available component scores x 100; NaN with fewer than min_components."""
    cols = [c for c in weights if weights[c] > 0]
    s = scores[cols]
    w = pd.Series(weights)[cols]
    available = s.notna()
    weighted = (s.fillna(0) * w).sum(axis=1)
    total = (available * w).sum(axis=1)
    index = 100 * weighted / total.replace(0, np.nan)
    return index.where(available.sum(axis=1) >= min_components)


def main_reason(scores: pd.DataFrame, weights: dict, index: pd.Series) -> pd.Series:
    """The component with the largest weighted contribution ('none' when the index is 0)."""
    cols = [c for c in weights if weights[c] > 0]
    contrib = scores[cols].mul(pd.Series(weights)[cols]).fillna(-1)
    reason = contrib.idxmax(axis=1).map(REASONS)
    reason[index.fillna(0) <= 0] = "none"
    reason[index.isna()] = ""
    return reason


def relative_bands(index: pd.Series, cuts=(0.70, 0.90)) -> pd.Series:
    lo, hi = index.quantile(cuts[0]), index.quantile(cuts[1])
    band = pd.Series("Low", index=index.index)
    band[index >= lo] = "Medium"
    band[index >= hi] = "High"
    band[index.isna()] = ""
    return band


def absolute_bands(comp: pd.DataFrame, cutoffs: dict) -> pd.Series:
    """High if any raw component reaches its High cut-off, or two reach their Medium cut-off;
    Medium if one reaches its Medium cut-off; else Low."""
    med = pd.DataFrame({c: comp[c] >= cutoffs[c][0] for c in cutoffs})
    high = pd.DataFrame({c: comp[c] >= cutoffs[c][1] for c in cutoffs})
    band = pd.Series("Low", index=comp.index)
    band[med.any(axis=1)] = "Medium"
    band[high.any(axis=1) | (med.sum(axis=1) >= 2)] = "High"
    return band


def build(readings: pd.DataFrame, cfg: SafetyConfig, incidents_csv: str | None = None) -> pd.DataFrame:
    """Components, scores, index, both bands and the main reason for every reading."""
    comp = compute_components(readings, cfg)
    scores = percentile_scores(comp)
    comp = comp.join(scores.add_suffix("_score"))
    comp["risk_index"] = risk_index(scores, cfg.weights, cfg.min_components)
    comp["band_relative"] = relative_bands(comp["risk_index"], cfg.relative_bands)
    comp["band_absolute"] = absolute_bands(comp, cfg.absolute_cutoffs)
    comp["main_reason"] = main_reason(scores, cfg.weights, comp["risk_index"])
    if incidents_csv is not None:
        comp = comp.join(incident_context(comp, readings, incidents_csv, cfg.incident_radius_km))
        if cfg.use_incidents:   # optional extra component; off by default
            inc_score = (comp["incident_count"] > 0).astype(float).where(comp["incident_count"].notna())
            scores = scores.assign(incident=inc_score)
            weights = {**cfg.weights, "incident": cfg.weights.get("incident", 1.0)}
            comp["risk_index_with_incidents"] = risk_index(scores, weights, cfg.min_components)
    return comp
