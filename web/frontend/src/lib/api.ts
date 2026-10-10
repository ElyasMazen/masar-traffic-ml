export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type RiskLevel = "Low" | "Medium" | "High";

export type RoadOption = {
  road_key: string;
  display_name: string;
  historical_crashes: number;
  risk_level: RiskLevel;
  risk_score: number;
  safety_score: number;
  avg_traffic: number;
  latitude: number | null;
  longitude: number | null;
};

export type MapRoad = RoadOption;

export type SegmentRoad = {
  segment_id: string;
  road_key: string;
  display_name: string;
  roadway_name: string;
  from_street: string | null;
  to_street: string | null;
  direction: string | null;
  avg_traffic: number;
  historical_crashes: number;
  observations: number;
  risk_score: number;
  safety_score: number;
  risk_level: RiskLevel;
  latitude: number | null;
  longitude: number | null;
};

export type Summary = {
  roads: number;
  road_segments: number;
  roads_with_accident_history: number;
  historical_crashes: number;
  average_traffic: number;
  high_risk_roads: number;
  model_available: boolean;
  model_metrics: {
    model: string;
    split: string;
    MAE: number;
    RMSE: number;
    R2: number;
  };
};

export type ForecastPoint = {
  datetime: string;
  hour: number;
  traffic: number;
  level: RiskLevel;
};

export type ExplanationFactor = {
  title: string;
  detail: string;
  impact: "up" | "down" | "neutral" | "confidence" | "caution";
};

export type GovernmentAction = {
  type: string;
  title: string;
  detail: string;
};

export type Simulation = {
  road: {
    road_key: string;
    display_name: string;
    latitude: number | null;
    longitude: number | null;
    historical_crashes: number;
    persons_injured: number;
    persons_killed: number;
    peak_hour: number | null;
  };
  datetime: string;
  predicted_traffic: number;
  traffic_level: RiskLevel;
  historical_range: {
    lower: number;
    upper: number;
    observations: number;
  };
  accident_risk: {
    score: number;
    level: RiskLevel;
    historical_crashes_same_weekday_hour: number;
    persons_injured_same_weekday_hour: number;
    persons_killed_same_weekday_hour: number;
    method: string;
  };
  road_safety: {
    score: number;
    level: string;
    note: string;
  };
  explanation: ExplanationFactor[];
  prediction_confidence: {
    level: "High" | "Medium" | "Low";
    detail: string;
    observations: number;
  };
  government_actions: GovernmentAction[];
  forecast: ForecastPoint[];
  model_mode: "machine_learning" | "historical_fallback";
  notice: string;
};

export type RoadInsights = {
  road_key: string;
  display_name: string;
  safety_score: number;
  safety_level: string;
  risk_score: number;
  risk_level: RiskLevel;
  historical_crashes: number;
  persons_injured: number;
  persons_killed: number;
  peak_hour: number | null;
  accident_time: {
    peak_window_start: number;
    peak_window_end: number;
    peak_window_crashes: number;
    top_day_name: string;
    weekend_share_pct: number;
  } | null;
  hourly_accidents: { hour: number; crashes: number }[];
  day_accidents: { day: string; day_of_week: number; crashes: number }[];
  top_causes: { cause: string; crashes: number; share_pct: number }[];
  data_coverage: {
    traffic_observations: number;
    first_traffic_observation: string | null;
    last_traffic_observation: string | null;
    first_crash: string | null;
    last_crash: string | null;
  };
  score_note: string;
};

export type ScenarioKey =
  | "rush_hour"
  | "accident"
  | "partial_closure"
  | "full_closure"
  | "special_event"
  | "signal_optimization";

export type ScenarioResult = {
  scenario: ScenarioKey;
  scenario_label: string;
  baseline: {
    predicted_traffic: number;
    traffic_level: RiskLevel;
    risk_score: number;
    pressure_index: number;
  };
  after: {
    estimated_segment_flow: number;
    traffic_level: RiskLevel;
    risk_score: number;
    pressure_index: number;
    network_disruption: string;
    nearby_pressure_pct: number;
  };
  difference: {
    segment_flow_pct: number;
    risk_points: number;
    pressure_index_points: number;
  };
  assumption: string;
  recommended_actions: string[];
  disclaimer: string;
};

export type PoliceRoadState = {
  road_key: string;
  display_name: string;
  predicted_traffic: number;
  traffic_level: RiskLevel;
  risk_score: number;
  risk_level: RiskLevel;
  historical_crashes_same_weekday_hour: number;
  persons_injured_same_weekday_hour: number;
  persons_killed_same_weekday_hour: number;
  alert_level: "Normal" | "Watch" | "Traffic alert" | "Safety alert" | "Critical";
};

export type PoliceAlert = PoliceRoadState & {
  severity_rank: number;
  message: string;
};

export type PoliceSnapshot = {
  datetime: string;
  date: string;
  hour: number;
  summary: {
    roads_modeled: number;
    high_traffic_roads: number;
    medium_traffic_roads: number;
    critical_alerts: number;
  };
  roads: PoliceRoadState[];
  alerts: PoliceAlert[];
  model_mode: "machine_learning" | "historical_fallback";
  notice: string;
};

export type PriorityRoad = {
  road_key: string;
  display_name: string;
  priority_score: number;
  priority_level: RiskLevel;
  avg_traffic: number;
  historical_crashes: number;
  persons_injured: number;
  persons_killed: number;
  risk_score: number;
  risk_level: RiskLevel;
  latitude: number | null;
  longitude: number | null;
};

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

export const api = {
  summary: () => getJson<Summary>(`${API_BASE}/api/summary`),
  roads: (search = "", limit = 30) =>
    getJson<RoadOption[]>(
      `${API_BASE}/api/roads?search=${encodeURIComponent(search)}&limit=${limit}`,
    ),
  mapRoads: () => getJson<MapRoad[]>(`${API_BASE}/api/roads/map`),
  segments: () => getJson<SegmentRoad[]>(`${API_BASE}/api/segments`),
  roadInsights: (roadKey: string) =>
    getJson<RoadInsights>(
      `${API_BASE}/api/roads/${encodeURIComponent(roadKey)}/insights`,
    ),
  hotspots: (limit = 8) =>
    getJson<MapRoad[]>(`${API_BASE}/api/hotspots?limit=${limit}`),
  priorityRoads: (limit = 8) =>
    getJson<PriorityRoad[]>(`${API_BASE}/api/government/priority?limit=${limit}`),
  policeSnapshot: async (date: string, hour: number) => {
    const res = await fetch(`${API_BASE}/api/police/snapshot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date, hour }),
    });
    if (!res.ok) throw new Error(`Police simulation failed: ${res.status}`);
    return (await res.json()) as PoliceSnapshot;
  },
  simulate: async (road_key: string, date: string, hour: number) => {
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ road_key, date, hour }),
    });
    if (!res.ok) throw new Error(`Simulation failed: ${res.status}`);
    return (await res.json()) as Simulation;
  },
  scenario: async (
    road_key: string,
    date: string,
    hour: number,
    scenario: ScenarioKey,
  ) => {
    const res = await fetch(`${API_BASE}/api/scenario`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ road_key, date, hour, scenario }),
    });
    if (!res.ok) throw new Error(`Scenario failed: ${res.status}`);
    return (await res.json()) as ScenarioResult;
  },
};
