from __future__ import annotations

import json
from datetime import date as Date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import JeddahSimulationRequest, PoliceSimulationRequest, ScenarioRequest, SimulationRequest
from .services.data_service import get_data_service
from .services.jeddah_service import JeddahService, get_jeddah_service
from .services.simulator import police_snapshot, simulate, simulate_scenario


BASE_DIR = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="Smart City Traffic Intelligence API",
    version="2.0.0",
    description="Historical traffic simulation, road-safety intelligence, scenario planning, and government decision support for NYC.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    ds = get_data_service()
    return {
        "status": "ok",
        "model_available": ds.model_available,
        "model_load_error": ds.model_load_error,
    }


@app.get("/api/summary")
def summary():
    ds = get_data_service()
    catalog = ds.catalog
    metrics = json.loads(
        (BASE_DIR / "data" / "processed" / "model_metrics.json").read_text()
    )
    return {
        "roads": int(len(catalog)),
        "road_segments": int(len(ds.segment_catalog)),
        "roads_with_accident_history": int((catalog["historical_crashes"] > 0).sum()),
        "historical_crashes": int(catalog["historical_crashes"].sum()),
        "average_traffic": round(float(catalog["avg_traffic"].mean()), 1),
        "high_risk_roads": int((catalog["risk_level"] == "High").sum()),
        "model_available": ds.model_available,
        "model_metrics": metrics,
    }


@app.get("/api/roads")
def roads(
    search: str = Query(default="", max_length=80),
    limit: int = Query(default=25, ge=1, le=100),
):
    return get_data_service().search_roads(search, limit)


@app.get("/api/roads/map")
def road_map():
    return get_data_service().map_roads()


@app.get("/api/segments")
def segments():
    return get_data_service().segments()


@app.get("/api/roads/{road_key}/insights")
def road_insights(road_key: str):
    result = get_data_service().road_insights(road_key)
    if result is None:
        raise HTTPException(status_code=404, detail="Road not found")
    return result


@app.get("/api/roads/{road_key}")
def road_profile(road_key: str):
    road = get_data_service().road(road_key)
    if road is None:
        raise HTTPException(status_code=404, detail="Road not found")
    return road


@app.get("/api/hotspots")
def hotspots(limit: int = Query(default=12, ge=1, le=100)):
    ds = get_data_service()
    df = (
        ds.catalog.dropna(subset=["median_latitude", "median_longitude"])
        .sort_values(["base_risk_score", "historical_crashes"], ascending=False)
        .head(limit)
    )
    return [
        {
            "road_key": str(r.road_key),
            "display_name": str(r.display_name),
            "risk_score": round(float(r.base_risk_score), 1),
            "safety_score": round(100 - float(r.base_risk_score), 1),
            "risk_level": str(r.risk_level),
            "historical_crashes": int(r.historical_crashes),
            "persons_injured": int(r.persons_injured),
            "persons_killed": int(r.persons_killed),
            "latitude": float(r.median_latitude),
            "longitude": float(r.median_longitude),
        }
        for r in df.itertuples(index=False)
    ]


@app.get("/api/government/priority")
def government_priority(limit: int = Query(default=8, ge=1, le=50)):
    return get_data_service().priority_roads(limit)


@app.post("/api/simulate")
def simulation(payload: SimulationRequest):
    try:
        return simulate(
            get_data_service(),
            road_key=payload.road_key,
            sim_date=payload.date,
            hour=payload.hour,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Road not found")


def _jeddah() -> JeddahService:
    try:
        return get_jeddah_service()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Jeddah models or training snapshot not found. Run: python scripts\\build_jeddah.py",
        )


@app.get("/api/jeddah/summary")
def jeddah_summary():
    return _jeddah().summary()


@app.get("/api/jeddah/points")
def jeddah_points():
    return _jeddah().points()


@app.get("/api/jeddah/short-term")
def jeddah_short_term():
    return _jeddah().short_term


@app.get("/api/jeddah/levels")
def jeddah_levels(date: Date, hour: int = Query(ge=0, le=23)):
    return _jeddah().levels(date, hour)


@app.post("/api/jeddah/simulate")
def jeddah_simulation(payload: JeddahSimulationRequest):
    try:
        return _jeddah().simulate(payload.point_id, payload.date, payload.hour)
    except KeyError:
        raise HTTPException(status_code=404, detail="Point not found")


@app.post("/api/police/snapshot")
def police_simulation_snapshot(payload: PoliceSimulationRequest):
    return police_snapshot(
        get_data_service(),
        sim_date=payload.date,
        hour=payload.hour,
    )


@app.post("/api/scenario")
def scenario(payload: ScenarioRequest):
    try:
        return simulate_scenario(
            get_data_service(),
            road_key=payload.road_key,
            sim_date=payload.date,
            hour=payload.hour,
            scenario=payload.scenario,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Road not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
