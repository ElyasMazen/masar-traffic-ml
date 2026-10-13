import { API_BASE, type RiskLevel } from "./api";

export type JeddahPoint = {
  point_id: string;
  road: string;
  point_index: number;
  label: string;
  short_label: string;
  latitude: number;
  longitude: number;
  frc: string;
  free_flow_speed_kmh: number;
  free_flow_travel_time_s: number;
  weekday_avg_speed_kmh: number;
  weekday_peak_hour: number;
  weekday_peak_level: RiskLevel;
};

export type JeddahLevel = {
  point_id: string;
  speed_ratio: number;
  speed_kmh: number;
  traffic_level: RiskLevel;
};

export type ShortTermPoint = {
  point_id: string;
  latest_timestamp: string;
  forecast_for: string;
  latest_speed_ratio: number;
  persistence_level: RiskLevel;
  predicted_level: RiskLevel;
  model_probability: number;
  probabilities: Record<RiskLevel, number>;
};

export type HorizonComparison = {
  split: string;
  split_label: string;
  horizon_min: number;
  random_forest_macro_f1: number;
  persistence_macro_f1: number;
};

export type ShortTermSelection = {
  primary_split: string;
  horizon_min: number;
  random_forest_macro_f1: number;
  persistence_macro_f1: number;
  margin: number;
  beats_persistence: boolean;
  rule: string;
  sentence: string;
  f1_noise: number;
  horizons_not_beating_persistence: number[];
  comparison: HorizonComparison[];
};

export type ShortTerm = {
  horizon_min: number;
  model: string;
  latest_timestamp: string;
  selected: ShortTermSelection;
  points: ShortTermPoint[];
};

/** "all" test rows, or only the weekday / weekend rows of a test that holds both. */
export type Subset = "all" | "weekday" | "weekend";

export type ModelPeriod = {
  train_start: string;
  train_end: string;
  note: string;
};

export type ShortTermResult = {
  split: string;
  horizon_min: number;
  method: "persistence" | "historical" | "random_forest";
  subset: Subset;
  test_daytypes: string;
  n: number;
  accuracy: number;
  macro_f1: number;
  recall_Low: number | null;
  recall_Medium: number | null;
  recall_High: number | null;
  support_High: number;
};

export type TimeOfDayResult = {
  split: string;
  method: string;
  subset: Subset;
  test_daytypes: string;
  n: number;
  mae_speed_kmh: number;
  accuracy: number;
  macro_f1: number;
  recall_High: number | null;
};

export type SplitInfo = {
  split: string;
  label: string;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  test_daytypes: string;
  test_weekend_days: string[];
  test_weekdays: string[];
};

export type JeddahSummary = {
  data: {
    source: string;
    rows: number;
    points: number;
    first_reading: string;
    last_reading: string;
    days: number;
    day_names: string[];
    trained_through: string;
    loaded_at: string;
    weekdays: string[];
    weekend_days: string[];
  };
  models: {
    evaluated: ModelPeriod & { split: string; test_start: string; test_end: string };
    final: ModelPeriod & { rows: number };
  };
  splits: SplitInfo[];
  short_term: {
    methods: Record<string, string>;
    selected: ShortTermSelection;
    results: ShortTermResult[];
  };
  time_of_day: {
    methods: Record<string, string>;
    selected: { method: string; label: string; rule: string; mae_speed_kmh: number; accuracy: number };
    results: TimeOfDayResult[];
  };
};

export type JeddahForecastPoint = {
  datetime: string;
  hour: number;
  speed_kmh: number;
  speed_ratio: number;
  level: RiskLevel;
};

export type JeddahSimulation = {
  point: JeddahPoint;
  datetime: string;
  day_type: "weekday" | "weekend";
  predicted_speed_kmh: number;
  predicted_speed_ratio: number;
  free_flow_speed_kmh: number;
  traffic_level: RiskLevel;
  expected_travel_time_s: number;
  free_flow_travel_time_s: number;
  historical_range: {
    lower_speed_kmh: number | null;
    upper_speed_kmh: number | null;
    days_observed: number;
  };
  /** How many days of readings back this point / hour / day type. A count, not a confidence. */
  data_coverage: {
    days_observed: number;
    summary: string;
    note: string;
  };
  method: { key: string; label: string };
  forecast: JeddahForecastPoint[];
  short_term: ShortTermPoint | null;
  short_term_horizon_min: number;
};

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

export const jeddahApi = {
  summary: () => getJson<JeddahSummary>(`${API_BASE}/api/jeddah/summary`),
  points: () => getJson<JeddahPoint[]>(`${API_BASE}/api/jeddah/points`),
  shortTerm: () => getJson<ShortTerm>(`${API_BASE}/api/jeddah/short-term`),
  levels: (date: string, hour: number) =>
    getJson<JeddahLevel[]>(`${API_BASE}/api/jeddah/levels?date=${date}&hour=${hour}`),
  simulate: async (point_id: string, date: string, hour: number) => {
    const res = await fetch(`${API_BASE}/api/jeddah/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point_id, date, hour }),
    });
    if (!res.ok) throw new Error(`Simulation failed: ${res.status}`);
    return (await res.json()) as JeddahSimulation;
  },
};
