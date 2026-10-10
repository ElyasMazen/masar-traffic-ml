from __future__ import annotations

from datetime import date, datetime, timedelta
import math

import pandas as pd

from .data_service import DataService


def _traffic_level(value: float, road: dict) -> str:
    low = float(road.get("traffic_low_threshold") or road.get("median_traffic") or 0)
    high = float(road.get("traffic_high_threshold") or road.get("avg_traffic") or 0)
    if value < low:
        return "Low"
    if value < high:
        return "Medium"
    return "High"


def _model_features(ds: DataService, road_key: str, dt: datetime) -> pd.DataFrame:
    bundle = ds.model_bundle
    if bundle is None:
        raise RuntimeError("ML model is not loaded")

    hour = dt.hour
    dow = dt.weekday()
    month = dt.month

    road_mean = float(bundle["road_mean"].get(road_key, bundle["global_mean"]))
    road_hour_mean = float(
        bundle["road_hour_mean"].get(f"{road_key}||{hour}", road_mean)
    )
    road_dow_hour_mean = float(
        bundle["road_dow_hour_mean"].get(
            f"{road_key}||{dow}||{hour}", road_hour_mean
        )
    )

    row = {
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "is_weekend": int(dow >= 5),
        "road_mean": road_mean,
        "road_hour_mean": road_hour_mean,
        "road_dow_hour_mean": road_dow_hour_mean,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
        "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
    }
    return pd.DataFrame([row], columns=bundle["features"])


def _predict_traffic(ds: DataService, road_key: str, dt: datetime) -> float:
    if ds.model_available:
        frame = _model_features(ds, road_key, dt)
        value = float(ds.model_bundle["model"].predict(frame)[0])
        return max(0.0, value)

    # Robust fallback if a local scikit-learn version cannot deserialize the model.
    pattern = ds.pattern_lookup.get((road_key, dt.month, dt.weekday(), dt.hour))
    if pattern:
        return max(0.0, float(pattern["mean"]))
    hourly = ds.road_hour_lookup.get((road_key, dt.weekday(), dt.hour))
    if hourly:
        return max(0.0, float(hourly["mean"]))
    road = ds.road(road_key) or {}
    return max(0.0, float(road.get("avg_traffic") or 0))


def _risk(ds: DataService, road_key: str, dt: datetime, road: dict) -> dict:
    key = (road_key, dt.weekday(), dt.hour)
    stats = ds.accident_lookup.get(
        key,
        {"historical_crashes": 0, "persons_injured": 0, "persons_killed": 0},
    )
    max_hour = max(1, int(ds.max_crashes_by_road.get(road_key, 1)))
    time_intensity = min(1.0, stats["historical_crashes"] / max_hour)

    base = float(road.get("base_risk_score") or 0)
    score = round(min(100.0, base * 0.70 + time_intensity * 100 * 0.30), 1)
    if score < 35:
        level = "Low"
    elif score < 65:
        level = "Medium"
    else:
        level = "High"

    return {
        "score": score,
        "level": level,
        "historical_crashes_same_weekday_hour": stats["historical_crashes"],
        "persons_injured_same_weekday_hour": stats["persons_injured"],
        "persons_killed_same_weekday_hour": stats["persons_killed"],
        "method": "Historical risk indicator — not a probability of a crash.",
    }


def _prediction_confidence(pattern: dict | None) -> dict:
    observations = int(pattern.get("observations", 0)) if pattern else 0
    if observations >= 8:
        return {
            "level": "High",
            "detail": "We have several similar historical records for this road and time.",
            "observations": observations,
        }
    if observations >= 3:
        return {
            "level": "Medium",
            "detail": "We have some similar historical records for this road and time.",
            "observations": observations,
        }
    return {
        "level": "Low",
        "detail": "There is limited historical data for this exact road and time.",
        "observations": observations,
    }


def _hour_label(hour: int) -> str:
    hour = int(hour) % 24
    suffix = "AM" if hour < 12 else "PM"
    display = hour % 12 or 12
    return f"{display} {suffix}"


def _explain_prediction(ds: DataService, road_key: str, dt: datetime, road: dict, prediction: float, pattern: dict | None) -> list[dict]:
    factors: list[dict] = []
    road_avg = float(road.get("avg_traffic") or prediction or 1)
    delta_pct = ((prediction - road_avg) / max(road_avg, 1)) * 100

    if abs(delta_pct) < 5:
        traffic_detail = "Traffic at this time is usually close to normal for this road."
        traffic_impact = "neutral"
    elif delta_pct > 0:
        traffic_detail = f"Traffic is usually about {abs(delta_pct):.0f}% higher than normal on this road at this time."
        traffic_impact = "up"
    else:
        traffic_detail = f"Traffic is usually about {abs(delta_pct):.0f}% lower than normal on this road at this time."
        traffic_impact = "down"

    factors.append({
        "title": "Traffic at this time",
        "detail": traffic_detail,
        "impact": traffic_impact,
    })

    peak_hour = road.get("peak_hour")
    if peak_hour is not None:
        distance = min((dt.hour - int(peak_hour)) % 24, (int(peak_hour) - dt.hour) % 24)
        if distance <= 1:
            detail = f"This time is close to the road's busiest period, around {_hour_label(int(peak_hour))}."
            impact = "up"
        else:
            detail = f"This road is usually busiest around {_hour_label(int(peak_hour))}."
            impact = "neutral"
        factors.append({
            "title": "Busiest time",
            "detail": detail,
            "impact": impact,
        })

    factors.append({
        "title": "Day of the week",
        "detail": (
            "You selected a weekend. Traffic usually follows weekend travel patterns."
            if dt.weekday() >= 5
            else "You selected a weekday. Traffic usually follows normal workday patterns."
        ),
        "impact": "neutral",
    })

    return factors


def _government_actions(traffic_level: str, risk: dict, insights: dict | None) -> list[dict]:
    actions: list[dict] = []

    if traffic_level == "High":
        actions.append({
            "type": "traffic",
            "title": "Prepare for heavy traffic",
            "detail": "Consider having traffic officers monitor the road, manage nearby intersections, or redirect cars if congestion builds.",
        })
    elif traffic_level == "Medium":
        actions.append({
            "type": "traffic",
            "title": "Keep an eye on traffic",
            "detail": "Traffic is busier than usual. Monitor the road and be ready to step in if congestion increases.",
        })

    if risk["level"] == "High":
        actions.append({
            "type": "response",
            "title": "Keep accident-response teams nearby",
            "detail": "This road has a high crash history at this time. Keeping authorized response teams nearby could help reduce response time.",
        })

    if insights and insights.get("top_causes"):
        top = str(insights["top_causes"][0]["cause"])
        top_upper = top.upper()
        if "SPEED" in top_upper:
            detail = "Speed is one of the main recorded crash factors here. Consider speed warnings, visible monitoring, or enforcement during higher-risk times."
        elif "ATTENTION" in top_upper or "DISTRACT" in top_upper:
            detail = "Driver inattention is one of the main recorded crash factors here. Clear warnings and visible monitoring may help drivers stay alert."
        elif "YIELD" in top_upper:
            detail = "Failure to yield is one of the main recorded crash factors here. It may help to review signs, visibility, and intersection control."
        else:
            detail = f"A common recorded crash factor here is '{top}'. This can help focus safety checks and road-safety measures."
        actions.append({
            "type": "safety",
            "title": "Focus on the main crash cause",
            "detail": detail,
        })

    if traffic_level == "High" and risk["level"] == "High":
        actions.insert(0, {
            "type": "priority",
            "title": "This road needs extra attention",
            "detail": "Heavy traffic and a high crash history happen at the same time. It may be useful to monitor this road closely and prepare a coordinated response.",
        })

    if not actions:
        actions.append({
            "type": "monitor",
            "title": "No special action needed",
            "detail": "The selected road and time do not show a high traffic or crash-risk warning. Normal monitoring is enough.",
        })

    return actions


def simulate(ds: DataService, road_key: str, sim_date: date, hour: int) -> dict:
    road = ds.road(road_key)
    if road is None:
        raise KeyError(road_key)

    dt = datetime.combine(sim_date, datetime.min.time()).replace(hour=hour)
    prediction = _predict_traffic(ds, road_key, dt)
    pattern = ds.pattern_lookup.get((road_key, dt.month, dt.weekday(), dt.hour))

    if pattern:
        lower = max(0.0, pattern["p25"])
        upper = max(lower, pattern["p75"])
        observations = pattern["observations"]
    else:
        spread = max(30.0, prediction * 0.22)
        lower = max(0.0, prediction - spread)
        upper = prediction + spread
        observations = 0

    risk = _risk(ds, road_key, dt, road)
    insights = ds.road_insights(road_key)
    safety_score = round(100 - risk["score"], 1)

    result = {
        "road": {
            "road_key": road_key,
            "display_name": road.get("display_name") or road_key,
            "latitude": road.get("median_latitude"),
            "longitude": road.get("median_longitude"),
            "historical_crashes": int(road.get("historical_crashes") or 0),
            "persons_injured": int(road.get("persons_injured") or 0),
            "persons_killed": int(road.get("persons_killed") or 0),
            "peak_hour": None if road.get("peak_hour") is None else int(road["peak_hour"]),
        },
        "datetime": dt.isoformat(),
        "predicted_traffic": round(prediction),
        "traffic_level": _traffic_level(prediction, road),
        "historical_range": {
            "lower": round(lower),
            "upper": round(upper),
            "observations": observations,
        },
        "accident_risk": risk,
        "road_safety": {
            "score": safety_score,
            "level": "Safer" if safety_score >= 70 else "Moderate" if safety_score >= 45 else "Needs attention",
            "note": "Higher is safer. This is a historical indicator, not a guarantee of future safety.",
        },
        "explanation": _explain_prediction(ds, road_key, dt, road, prediction, pattern),
        "prediction_confidence": _prediction_confidence(pattern),
        "government_actions": [],
        "forecast": [],
        "model_mode": "machine_learning" if ds.model_available else "historical_fallback",
        "notice": "This is a historical-pattern simulation based on NYC traffic counts, not live traffic.",
    }
    result["government_actions"] = _government_actions(result["traffic_level"], risk, insights)

    for offset in range(0, 12):
        future_dt = dt + timedelta(hours=offset)
        value = _predict_traffic(ds, road_key, future_dt)
        result["forecast"].append(
            {
                "datetime": future_dt.isoformat(),
                "hour": future_dt.hour,
                "traffic": round(value),
                "level": _traffic_level(value, road),
            }
        )
    return result


SCENARIOS = {
    "accident": {
        "label": "Accident occurs",
        "flow_factor": 0.72,
        "pressure_index": 140,
        "risk_delta": 18,
        "nearby_pressure_pct": 35,
        "disruption": "High",
        "assumption": "Planning assumption: an incident reduces effective segment throughput and increases local network pressure.",
    },
    "partial_closure": {
        "label": "Partial road closure",
        "flow_factor": 0.55,
        "pressure_index": 155,
        "risk_delta": 10,
        "nearby_pressure_pct": 45,
        "disruption": "High",
        "assumption": "Planning assumption: reduced lane availability lowers throughput and displaces demand to nearby streets.",
    },
    "full_closure": {
        "label": "Full road closure",
        "flow_factor": 0.08,
        "pressure_index": 190,
        "risk_delta": 12,
        "nearby_pressure_pct": 70,
        "disruption": "Severe",
        "assumption": "Planning assumption: the selected segment is effectively unavailable and demand is displaced. Nearby routing is not microsimulated.",
    },
    "special_event": {
        "label": "Major event nearby",
        "flow_factor": 1.20,
        "pressure_index": 125,
        "risk_delta": 8,
        "nearby_pressure_pct": 20,
        "disruption": "Moderate",
        "assumption": "Planning assumption: a major event raises local travel demand around the selected period.",
    },
    "signal_optimization": {
        "label": "Signal timing improvement",
        "flow_factor": 1.00,
        "pressure_index": 88,
        "risk_delta": -3,
        "nearby_pressure_pct": -5,
        "disruption": "Improved",
        "assumption": "Illustrative planning assumption only: optimized signal timing reduces congestion pressure. Signal delay is not directly modeled by this dataset.",
    },
}


def simulate_scenario(ds: DataService, road_key: str, sim_date: date, hour: int, scenario: str) -> dict:
    baseline = simulate(ds, road_key, sim_date, hour)
    road = ds.road(road_key)
    if road is None:
        raise KeyError(road_key)

    baseline_flow = float(baseline["predicted_traffic"])
    baseline_risk = float(baseline["accident_risk"]["score"])

    if scenario == "rush_hour":
        peak_hour = int(road.get("peak_hour") if road.get("peak_hour") is not None else hour)
        peak_dt = datetime.combine(sim_date, datetime.min.time()).replace(hour=peak_hour)
        after_flow = _predict_traffic(ds, road_key, peak_dt)
        ratio = after_flow / max(baseline_flow, 1)
        pressure_index = round(max(100, min(180, 100 * ratio)))
        risk_after = _risk(ds, road_key, peak_dt, road)["score"]
        spec = {
            "label": f"Rush hour ({peak_hour:02d}:00)",
            "nearby_pressure_pct": round(max(0, (ratio - 1) * 100)),
            "disruption": "High" if pressure_index >= 125 else "Moderate",
            "assumption": "Rush-hour comparison uses the same ML/historical model at this road's observed peak hour.",
        }
    else:
        if scenario not in SCENARIOS:
            raise ValueError("Unknown scenario")
        spec = SCENARIOS[scenario]
        after_flow = baseline_flow * float(spec["flow_factor"])
        pressure_index = int(spec["pressure_index"])
        risk_after = max(0.0, min(100.0, baseline_risk + float(spec["risk_delta"])))

    if scenario == "rush_hour":
        label = spec["label"]
    else:
        label = str(spec["label"])

    flow_delta_pct = round(((after_flow - baseline_flow) / max(baseline_flow, 1)) * 100, 1)
    risk_delta = round(float(risk_after) - baseline_risk, 1)

    actions = []
    if pressure_index >= 140:
        actions.append("Prepare traffic diversion and field traffic-control options before the scenario window.")
    elif pressure_index >= 120:
        actions.append("Increase monitoring and prepare manual traffic management if queues build.")
    if risk_after >= 65:
        actions.append("Increase incident-response readiness near the corridor during the scenario window.")
        actions.append("For a Saudi deployment, this can support positioning Najm or other authorized accident-response services nearby.")
    if scenario in {"partial_closure", "full_closure", "accident"}:
        actions.append("Notify road users early and consider alternate-route messaging to reduce secondary congestion.")
    if scenario == "signal_optimization":
        actions.append("Validate any signal-timing change with engineering review and field measurements before operational use.")

    return {
        "scenario": scenario,
        "scenario_label": label,
        "baseline": {
            "predicted_traffic": round(baseline_flow),
            "traffic_level": baseline["traffic_level"],
            "risk_score": baseline_risk,
            "pressure_index": 100,
        },
        "after": {
            "estimated_segment_flow": round(after_flow),
            "traffic_level": _traffic_level(after_flow, road),
            "risk_score": round(float(risk_after), 1),
            "pressure_index": pressure_index,
            "network_disruption": spec["disruption"],
            "nearby_pressure_pct": spec["nearby_pressure_pct"],
        },
        "difference": {
            "segment_flow_pct": flow_delta_pct,
            "risk_points": risk_delta,
            "pressure_index_points": pressure_index - 100,
        },
        "assumption": spec["assumption"],
        "recommended_actions": actions,
        "disclaimer": "Scenario Lab is a planning what-if tool. Except for the rush-hour comparison, scenario effects are transparent assumptions, not causal traffic microsimulation.",
    }



def _batch_predict_traffic(ds: DataService, dt: datetime) -> dict[str, float]:
    """Predict one simulated hour for every modeled road in one model call."""
    road_keys = ds.catalog["road_key"].astype(str).tolist()

    if ds.model_available:
        bundle = ds.model_bundle
        rows = []
        hour = dt.hour
        dow = dt.weekday()
        month = dt.month
        for road_key in road_keys:
            road_mean = float(bundle["road_mean"].get(road_key, bundle["global_mean"]))
            road_hour_mean = float(
                bundle["road_hour_mean"].get(f"{road_key}||{hour}", road_mean)
            )
            road_dow_hour_mean = float(
                bundle["road_dow_hour_mean"].get(
                    f"{road_key}||{dow}||{hour}", road_hour_mean
                )
            )
            rows.append({
                "hour": hour,
                "day_of_week": dow,
                "month": month,
                "is_weekend": int(dow >= 5),
                "road_mean": road_mean,
                "road_hour_mean": road_hour_mean,
                "road_dow_hour_mean": road_dow_hour_mean,
                "hour_sin": math.sin(2 * math.pi * hour / 24),
                "hour_cos": math.cos(2 * math.pi * hour / 24),
                "dow_sin": math.sin(2 * math.pi * dow / 7),
                "dow_cos": math.cos(2 * math.pi * dow / 7),
                "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
                "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
            })

        frame = pd.DataFrame(rows, columns=bundle["features"])
        values = bundle["model"].predict(frame)
        return {
            road_key: max(0.0, float(value))
            for road_key, value in zip(road_keys, values)
        }

    # Historical fallback if the local model cannot be loaded.
    predictions: dict[str, float] = {}
    for road_key in road_keys:
        pattern = ds.pattern_lookup.get((road_key, dt.month, dt.weekday(), dt.hour))
        if pattern:
            value = float(pattern["mean"])
        else:
            hourly = ds.road_hour_lookup.get((road_key, dt.weekday(), dt.hour))
            if hourly:
                value = float(hourly["mean"])
            else:
                road = ds.road(road_key) or {}
                value = float(road.get("avg_traffic") or 0)
        predictions[road_key] = max(0.0, value)
    return predictions


def police_snapshot(ds: DataService, sim_date: date, hour: int) -> dict:
    """
    City-wide planning playback for police/traffic authorities.

    This is deliberately a simulated clock based on historical traffic patterns,
    not a live feed. One snapshot colors every modeled road by expected traffic
    for the selected simulated hour and surfaces the strongest operational alerts.
    """
    dt = datetime.combine(sim_date, datetime.min.time()).replace(hour=hour)
    predictions = _batch_predict_traffic(ds, dt)

    road_states: list[dict] = []
    alerts: list[dict] = []
    high_count = 0
    medium_count = 0
    critical_count = 0

    for row in ds.catalog.itertuples(index=False):
        road_key = str(row.road_key)
        prediction = predictions.get(road_key, float(row.avg_traffic))
        road = row._asdict()
        traffic_level = _traffic_level(prediction, road)

        accident_stats = ds.accident_lookup.get(
            (road_key, dt.weekday(), dt.hour),
            {"historical_crashes": 0, "persons_injured": 0, "persons_killed": 0},
        )
        max_hour = max(1, int(ds.max_crashes_by_road.get(road_key, 1)))
        time_intensity = min(1.0, accident_stats["historical_crashes"] / max_hour)
        base_risk = float(getattr(row, "base_risk_score", 0) or 0)
        risk_score = round(min(100.0, base_risk * 0.70 + time_intensity * 100 * 0.30), 1)
        risk_level = "Low" if risk_score < 35 else "Medium" if risk_score < 65 else "High"

        if traffic_level == "High":
            high_count += 1
        elif traffic_level == "Medium":
            medium_count += 1

        reasons: list[str] = []
        severity_rank = 0
        alert_level = "Normal"

        if traffic_level == "High" and risk_level == "High":
            alert_level = "Critical"
            severity_rank = 4
            critical_count += 1
            reasons.append("High traffic and high historical crash risk")
        elif traffic_level == "High":
            alert_level = "Traffic alert"
            severity_rank = 3
            reasons.append("High traffic expected")
        elif risk_level == "High" and accident_stats["historical_crashes"] > 0:
            alert_level = "Safety alert"
            severity_rank = 3
            reasons.append("High historical crash risk at this time")
        elif traffic_level == "Medium" and risk_level in {"Medium", "High"}:
            alert_level = "Watch"
            severity_rank = 2
            reasons.append("Traffic and safety conditions need monitoring")

        display_name = str(getattr(row, "display_name", road_key))
        state = {
            "road_key": road_key,
            "display_name": display_name,
            "predicted_traffic": round(prediction),
            "traffic_level": traffic_level,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "historical_crashes_same_weekday_hour": int(accident_stats["historical_crashes"]),
            "persons_injured_same_weekday_hour": int(accident_stats["persons_injured"]),
            "persons_killed_same_weekday_hour": int(accident_stats["persons_killed"]),
            "alert_level": alert_level,
        }
        road_states.append(state)

        if severity_rank >= 2:
            alerts.append({
                **state,
                "severity_rank": severity_rank,
                "message": ". ".join(reasons),
            })

    alerts.sort(
        key=lambda item: (
            item["severity_rank"],
            item["predicted_traffic"],
            item["historical_crashes_same_weekday_hour"],
        ),
        reverse=True,
    )

    return {
        "datetime": dt.isoformat(),
        "date": sim_date.isoformat(),
        "hour": hour,
        "summary": {
            "roads_modeled": len(road_states),
            "high_traffic_roads": high_count,
            "medium_traffic_roads": medium_count,
            "critical_alerts": critical_count,
        },
        "roads": road_states,
        "alerts": alerts[:10],
        "model_mode": "machine_learning" if ds.model_available else "historical_fallback",
        "notice": (
            "Police Simulation is a planning playback based on historical NYC traffic and crash patterns. "
            "The clock is simulated; values are not live traffic or live accident reports."
        ),
    }
