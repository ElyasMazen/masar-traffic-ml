from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    road_key: str = Field(min_length=1)
    date: date
    hour: int = Field(ge=0, le=23)


class JeddahSimulationRequest(BaseModel):
    point_id: str = Field(min_length=1)
    date: date
    hour: int = Field(ge=0, le=23)


class PoliceSimulationRequest(BaseModel):
    date: date
    hour: int = Field(ge=0, le=23)


ScenarioKey = Literal[
    "rush_hour",
    "accident",
    "partial_closure",
    "full_closure",
    "special_event",
    "signal_optimization",
]


class ScenarioRequest(SimulationRequest):
    scenario: ScenarioKey


class RoadSummary(BaseModel):
    road_key: str
    display_name: str
    latitude: float | None = None
    longitude: float | None = None
    historical_crashes: int = 0
    persons_injured: int = 0
    persons_killed: int = 0
    avg_traffic: float = 0
    peak_hour: int | None = None
    base_risk_score: float = 0
    risk_level: str = "Low"
