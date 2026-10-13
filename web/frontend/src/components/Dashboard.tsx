"use client";

import Image from "next/image";
import { useCallback, useEffect, useMemo, useState } from "react";
import CitySelect, { type City } from "./CitySelect";
import ForecastChart from "./ForecastChart";
import JeddahWorkspace from "./JeddahWorkspace";
import TrafficMap from "./TrafficMap";
import {
  api,
  type MapRoad,
  type RoadOption,
  type SegmentRoad,
  type Simulation,
} from "@/lib/api";
import { LOCALE, formatDateHour, formatHour, isoDate, levelClass } from "@/lib/format";

const today = isoDate(new Date());

export default function Dashboard() {
  const [city, setCity] = useState<City>("jeddah");
  const [mapRoads, setMapRoads] = useState<MapRoad[]>([]);
  const [segments, setSegments] = useState<SegmentRoad[]>([]);
  const [roadOptions, setRoadOptions] = useState<RoadOption[]>([]);
  const [selectedRoad, setSelectedRoad] = useState<RoadOption | null>(null);
  const [search, setSearch] = useState("");
  const [date, setDate] = useState(today);
  const [hour, setHour] = useState(17);
  const [simulation, setSimulation] = useState<Simulation | null>(null);
  const [loading, setLoading] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.mapRoads(), api.segments(), api.roads("", 12)])
      .then(([mapData, segmentData, roadsData]) => {
        setMapRoads(mapData);
        setSegments(segmentData);
        setRoadOptions(roadsData);
        if (roadsData[0]) {
          setSelectedRoad(roadsData[0]);
          setSearch(roadsData[0].display_name);
        }
      })
      .catch(() => setError("Could not connect to the backend."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      api.roads(search, 12).then(setRoadOptions).catch(() => undefined);
    }, 180);
    return () => clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    setSimulation(null);
  }, [selectedRoad, date, hour]);

  const onMapSelect = useCallback((road: MapRoad) => {
    setSelectedRoad(road);
    setSearch(road.display_name);
    setRoadOptions([]);
  }, []);

  async function runSimulation() {
    if (!selectedRoad) return;
    setSimulating(true);
    setError("");
    try {
      const result = await api.simulate(selectedRoad.road_key, date, hour);
      setSimulation(result);
      window.setTimeout(() => {
        document.getElementById("simulation-output")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 70);
    } catch {
      setError("Simulation failed. Make sure the backend is running.");
    } finally {
      setSimulating(false);
    }
  }

  const selectedMapRoad = useMemo(
    () => mapRoads.find((road) => road.road_key === selectedRoad?.road_key) ?? selectedRoad,
    [mapRoads, selectedRoad],
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <Image
            src="/masar-logo.png"
            alt="مسار"
            width={1231}
            height={902}
            priority
            className="brand-logo"
          />
          <div className="brand-subtitle">Traffic Intelligence Prototype</div>
        </div>
        <span className="prototype-badge">Prototype</span>
      </header>

      <section className="hero simple-hero">
        <div>
          <div className="eyebrow">MASAR · TRAFFIC INTELLIGENCE</div>
          <h1>Understand the road<br />before traffic builds.</h1>
          <p>
            Select a Jeddah road and time, then view a simple traffic prediction on an interactive map.
          </p>
        </div>
      </section>

      {city === "jeddah" ? (
        <JeddahWorkspace city={city} onCityChange={setCity} />
      ) : (
      <>
      <section className="workspace prototype-workspace">
        <aside className="control-card">
          <div className="section-kicker">TRAFFIC SIMULATION</div>
          <h2>Choose a road and time</h2>
          <p className="muted">Search by road name or click a colored road directly on the map.</p>

          <CitySelect city={city} onChange={setCity} />

          <label className="field-label">Road</label>
          <div className="search-wrap">
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search Broadway, Atlantic Ave..."
              className="field-input"
            />
            {search && roadOptions.length > 0 && (
              <div className="road-dropdown">
                {roadOptions.slice(0, 7).map((road) => (
                  <button
                    key={road.road_key}
                    className="road-option"
                    onClick={() => {
                      setSelectedRoad(road);
                      setSearch(road.display_name);
                      setRoadOptions([]);
                    }}
                  >
                    <span>
                      <strong>{road.display_name}</strong>
                      <small>{road.historical_crashes.toLocaleString(LOCALE)} historical crashes</small>
                    </span>
                    <span className={levelClass(road.risk_level)}>{road.risk_level}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="two-fields">
            <div>
              <label className="field-label">Date</label>
              <input
                type="date"
                value={date}
                onChange={(event) => setDate(event.target.value)}
                className="field-input"
              />
            </div>
            <div>
              <label className="field-label">Hour</label>
              <select
                value={hour}
                onChange={(event) => setHour(Number(event.target.value))}
                className="field-input"
              >
                {Array.from({ length: 24 }, (_, value) => (
                  <option key={value} value={value}>{formatHour(value)}</option>
                ))}
              </select>
            </div>
          </div>

          <button
            className="simulate-button"
            onClick={runSimulation}
            disabled={!selectedRoad || simulating}
          >
            {simulating ? "Running…" : "Simulate selected road"}
          </button>

          {selectedMapRoad && (
            <div className="selected-road-card">
              <div className="selected-road-top">
                <span>Selected road</span>
                <span className={levelClass(selectedMapRoad.risk_level)}>
                  {selectedMapRoad.risk_level} risk
                </span>
              </div>
              <strong>{selectedMapRoad.display_name}</strong>
              <div className="selected-road-stats">
                <span>{Math.round(selectedMapRoad.avg_traffic).toLocaleString(LOCALE)} avg traffic</span>
                <span>{selectedMapRoad.historical_crashes.toLocaleString(LOCALE)} crashes</span>
              </div>
              <div className="safety-mini-row">
                <span>Safety score</span>
                <strong>{Math.round(selectedMapRoad.safety_score ?? 100)}/100</strong>
              </div>
            </div>
          )}

          <div className="control-footnote">
            Prototype predictions use historical NYC data and are not live traffic measurements.
          </div>
        </aside>

        <div className="map-card">
          <div className="map-toolbar">
            <div>
              <div className="section-kicker">INTERACTIVE STREET MAP</div>
              <h2>Click the road itself</h2>
            </div>
            <div className="map-legend">
              <span><i className="dot-low" />Low risk</span>
              <span><i className="dot-medium" />Medium</span>
              <span><i className="dot-high" />High</span>
            </div>
          </div>
          <TrafficMap
            roads={mapRoads}
            segments={segments}
            selectedRoadKey={selectedRoad?.road_key ?? null}
            onSelect={onMapSelect}
          />
        </div>
      </section>

      {error && <div className="error-banner">{error}</div>}

      <section className="results-section simple-results" id="simulation-output">
        <div className="results-header">
          <div>
            <div className="section-kicker">SIMULATION OUTPUT</div>
            <h2>{simulation ? simulation.road.display_name : "Your result will appear here"}</h2>
          </div>
          {simulation && (
            <div className="result-time">
              {formatDateHour(simulation.datetime)}
            </div>
          )}
        </div>

        {!simulation ? (
          <div className="empty-state compact-empty">
            <div>
              <strong>Select a road, date and hour.</strong>
              <span>Then press “Simulate selected road” to see the prediction.</span>
            </div>
          </div>
        ) : (
          <>
            <div className="result-grid result-grid-five">
              <div className="result-card">
                <span>Expected traffic</span>
                <strong>{simulation.predicted_traffic.toLocaleString(LOCALE)}</strong>
                <small>estimated vehicles</small>
              </div>
              <div className="result-card">
                <span>Traffic level</span>
                <strong className={`result-level level-text-${simulation.traffic_level.toLowerCase()}`}>
                  {simulation.traffic_level}
                </strong>
                <small>{simulation.historical_range.lower.toLocaleString(LOCALE)}–{simulation.historical_range.upper.toLocaleString(LOCALE)} historical range</small>
              </div>
              <div className="result-card">
                <span>Accident risk</span>
                <strong className={`result-level level-text-${simulation.accident_risk.level.toLowerCase()}`}>
                  {simulation.accident_risk.score}/100
                </strong>
                <small>{simulation.accident_risk.level} historical risk</small>
              </div>
              <div className="result-card result-card-safety">
                <span>Road Safety Score</span>
                <strong>{simulation.road_safety.score}/100</strong>
                <small>Higher is safer</small>
              </div>
              <div className="result-card">
                <span>Similar past crashes</span>
                <strong>{simulation.accident_risk.historical_crashes_same_weekday_hour.toLocaleString(LOCALE)}</strong>
                <small>same road at a similar time</small>
              </div>
            </div>

            <div className="chart-card simple-chart-card">
              <div className="chart-header">
                <div>
                  <h3>12-hour traffic simulation</h3>
                  <p>Expected traffic after the selected hour.</p>
                </div>
                <span className="model-chip">Prototype prediction</span>
              </div>
              <ForecastChart data={simulation.forecast} />
            </div>
          </>
        )}
      </section>
      </>
      )}

      <footer className="simple-footer">
        <span>مسار · Prototype</span>
        <span>
          {city === "jeddah"
            ? "Jeddah TomTom traffic data collected for Masar"
            : "Historical New York City traffic data"}
        </span>
      </footer>

      {loading && city === "new-york" && <div className="loading-overlay">Loading prototype…</div>}
    </main>
  );
}
