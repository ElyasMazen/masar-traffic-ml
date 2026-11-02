import { API_BASE, type RiskLevel } from "./api";

/** Safety risk indicator - not accident prediction (see docs/SAFETY_INDICATOR.md). */
export type SafetyLevel = {
  point_id: string;
  day_type: "weekday" | "weekend";
  hour: number;
  /** 90th percentile of the index at this point, day type and hour (0-100). */
  score: number;
  band: RiskLevel;
  main_component: string;
  reason: string;
  /** Share of past readings at this hour that were Medium or High. */
  share_elevated: number;
  share_high: number;
  readings: number;
  days: number;
};

export type SafetyLatest = {
  point_id: string;
  reading_time: string;
  score: number;
  band: RiskLevel;
  main_component: string;
  reason: string;
  speed_kmh: number;
};

export type SafetySummary = {
  name: string;
  note: string;
  definition: string;
  score_for_date_hour: string;
  cutoffs: { medium: number; high: number };
};

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

export const safetyApi = {
  summary: () => getJson<SafetySummary>(`${API_BASE}/api/jeddah/safety/summary`),
  latest: () => getJson<SafetyLatest[]>(`${API_BASE}/api/jeddah/safety/latest`),
  levels: (date: string, hour: number) =>
    getJson<SafetyLevel[]>(`${API_BASE}/api/jeddah/safety/levels?date=${date}&hour=${hour}`),
};
