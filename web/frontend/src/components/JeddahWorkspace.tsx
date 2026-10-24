"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import CitySelect, { type City } from "./CitySelect";
import ForecastChart from "./ForecastChart";
import JeddahMap from "./JeddahMap";
import {
  formatClock,
  formatDateHour,
  formatDateRange,
  formatDateTime,
  formatHour,
  horizonTitle,
  isoDate,
  levelClass,
  ratioPercent,
  toDate,
} from "@/lib/format";
import {
  jeddahApi,
  type JeddahLevel,
  type JeddahPoint,
  type JeddahSimulation,
  type JeddahSummary,
  type ShortTerm,
  type ShortTermPoint,
  type Subset,
  type TimeOfDayResult,
} from "@/lib/jeddah";

const WEEKEND = [5, 6]; // Friday, Saturday (JS getDay)
const SUBSETS: Subset[] = ["all", "weekday", "weekend"];
const TIME_METHOD_SHORT: Record<string, string> = {
  time_model_hgb: "Time model",
  avg_point_hour: "Avg: point × hour",
  avg_point_hour_daytype: "Avg: point × hour × day type",
};

/** Today, or the next Sun-Thu working day when today is a Saudi weekend day. */
function defaultDate() {
  const day = new Date();
  while (WEEKEND.includes(day.getDay())) day.setDate(day.getDate() + 1);
  return isoDate(day);
}

function isWeekend(date: string) {
  return Boolean(date) && WEEKEND.includes(toDate(`${date}T12:00:00`).getDay());
}

function pct(value: number | null | undefined) {
  return value == null ? "n/a" : `${Math.round(value * 100)}%`;
}

/**
 * "+8 min (21 min instead of 13 min at free flow)". The delay is rounded once,
 * from the exact seconds; the travel time shown is free flow + that delay, so
 * the two parts of the text always agree.
 */
function delayText(travelS: number, freeFlowS: number) {
  const minutes = Math.max(travelS, freeFlowS) >= 120;
  const unit = minutes ? 60 : 1;
  const label = minutes ? "min" : "s";
  const delay = Math.round((travelS - freeFlowS) / unit);
  const free = Math.round(freeFlowS / unit);
  return {
    delta: `+${delay} ${label}`,
    detail: `(${free + delay} ${label} instead of ${free} ${label} at free flow)`,
  };
}

/** Random Forest level for latest reading + horizon, always labelled with the reading it starts from. */
function NextHourCard({ forecast, horizon }: { forecast: ShortTermPoint | null; horizon: number }) {
  return (
    <div className="result-card result-card-safety">
      <span>{horizonTitle(horizon)} · +{horizon} min from latest reading</span>
      {forecast ? (
        <>
          <strong className={`level-text-${forecast.predicted_level.toLowerCase()}`}>{forecast.predicted_level}</strong>
          <small>
            for {formatDateTime(forecast.forecast_for)} · from the {formatDateTime(forecast.latest_reading_time ?? forecast.latest_timestamp)} reading ·
            model probability {pct(forecast.model_probability)} · persistence says {forecast.persistence_level}
          </small>
        </>
      ) : (
        <>
          <strong>—</strong>
          <small>Not enough recent readings for this point</small>
        </>
      )}
    </div>
  );
}

export default function JeddahWorkspace({
  city,
  onCityChange,
}: {
  city: City;
  onCityChange: (city: City) => void;
}) {
  const [points, setPoints] = useState<JeddahPoint[]>([]);
  const [summary, setSummary] = useState<JeddahSummary | null>(null);
  const [shortTerm, setShortTerm] = useState<ShortTerm | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [date, setDate] = useState(defaultDate);
  const [hour, setHour] = useState(17);
  const [levels, setLevels] = useState<Record<string, JeddahLevel>>({});
  const [simulation, setSimulation] = useState<JeddahSimulation | null>(null);
  const [predictedAt, setPredictedAt] = useState(0); // when "Predict traffic" was pressed (ms)
  const [loading, setLoading] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([jeddahApi.points(), jeddahApi.summary(), jeddahApi.shortTerm()])
      .then(([pointData, summaryData, shortTermData]) => {
        setPoints(pointData);
        setSummary(summaryData);
        setShortTerm(shortTermData);
        // Start on the point that gets most congested on weekdays, so the demo shows something.
        const order = { High: 0, Medium: 1, Low: 2 };
        const busiest = [...pointData].sort((a, b) => order[a.weekday_peak_level] - order[b.weekday_peak_level])[0];
        setSelectedId(busiest?.point_id ?? null);
      })
      .catch(() =>
        setError("Could not load Jeddah data. Make sure the backend is running and scripts\\build_jeddah.py has been run."),
      )
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!date) return;
    let cancelled = false;
    jeddahApi
      .levels(date, hour)
      .then((rows) => {
        if (!cancelled) setLevels(Object.fromEntries(rows.map((row) => [row.point_id, row])));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [date, hour]);

  const selectPoint = useCallback((point: JeddahPoint) => {
    setSelectedId(point.point_id);
    setSimulation(null);
  }, []);

  async function runSimulation() {
    if (!selectedId) return;
    setSimulating(true);
    setError("");
    try {
      const result = await jeddahApi.simulate(selectedId, date, hour);
      setPredictedAt(Date.now());
      setSimulation(result);
      window.setTimeout(() => {
        document.getElementById("simulation-output")?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 70);
    } catch {
      setError("Simulation failed. Make sure the backend is running.");
    } finally {
      setSimulating(false);
    }
  }

  const selected = points.find((point) => point.point_id === selectedId) ?? null;
  const weekendSelected = isWeekend(date);

  const roads = useMemo(() => {
    const grouped = new Map<string, JeddahPoint[]>();
    for (const point of points) grouped.set(point.road, [...(grouped.get(point.road) ?? []), point]);
    return [...grouped.entries()];
  }, [points]);

  const data = summary?.data;
  const models = summary?.models;
  const trainedOn = data ? `Data: ${formatDateRange(data.first_reading, data.trained_through)}, ${data.days} days` : "";
  const dayCount = data?.weekdays ? `${data.weekdays.length} weekdays, ${data.weekend_days.length} weekend days` : "";
  const splitLabel = (name: string) => summary?.splits.find((s) => s.split === name)?.label ?? name;
  /** "Last 4 days (Sat 10 Oct – Tue 13 Oct)" -> "Last 4 days", plus which day type a row covers. */
  const testLabel = (split: string, subset: Subset) => {
    const short = splitLabel(split).split(" (")[0];
    if (subset !== "all") return `${short} · ${subset === "weekday" ? "weekdays" : "weekend"} only`;
    const days = summary?.splits.find((s) => s.split === split)?.test_daytypes ?? "";
    return `${short} (${days.replace("+", " + ")})`;
  };
  const horizon = shortTerm?.horizon_min;
  const selection = shortTerm?.selected;

  const shortRows = useMemo(() => {
    if (!summary) return [];
    const results = summary.short_term.results;
    const find = (split: string, h: number, subset: Subset, method: string) =>
      results.find((r) => r.split === split && r.horizon_min === h && r.subset === subset && r.method === method);
    return summary.splits.flatMap((sp) =>
      (summary.data.horizons_min ?? [10, 30, 60]).flatMap((h) =>
        SUBSETS.filter((subset) => find(sp.split, h, subset, "random_forest")).map((subset) => ({
          split: sp.split,
          horizon: h,
          subset,
          rf: find(sp.split, h, subset, "random_forest"),
          persistence: find(sp.split, h, subset, "persistence"),
          historical: find(sp.split, h, subset, "historical"),
        })),
      ),
    );
  }, [summary]);

  const timeRows = useMemo(() => {
    if (!summary) return [];
    const order = (r: TimeOfDayResult) =>
      summary.splits.findIndex((s) => s.split === r.split) * 10 + SUBSETS.indexOf(r.subset);
    return [...summary.time_of_day.results].sort((a, b) => order(a) - order(b));
  }, [summary]);

  // The +60 min card is "from the latest reading", not "for the selected time".
  // Show it beside the selected-time cards only when the two are close.
  const nearLatest =
    simulation && shortTerm
      ? Math.abs(toDate(simulation.datetime).getTime() - toDate(shortTerm.latest_timestamp).getTime()) <=
        2 * 60 * 60 * 1000
      : false;
  const selectedNext = shortTerm?.points.find((p) => p.point_id === selectedId) ?? null;

  // Readings are resampled to the training step (e.g. 30 min); say so wherever counts appear.
  const stepMin = data?.step_min ?? shortTerm?.step_min;
  const readingsText = data
    ? stepMin && data.rows_raw && data.rows_raw !== data.rows
      ? `${data.rows.toLocaleString("en-GB")} readings on a ${stepMin}-minute grid, from ${data.rows_raw.toLocaleString("en-GB")} collected`
      : `${data.rows.toLocaleString("en-GB")} TomTom readings${stepMin ? `, ${stepMin}-minute step` : ""}`
    : "";
  const latestReading = shortTerm?.points
    .map((p) => p.latest_reading_time ?? p.latest_timestamp)
    .sort()
    .at(-1);

  // The selected hour had fully passed when the user pressed Predict: what is shown is still a prediction.
  const selectedIsPast = simulation
    ? toDate(simulation.datetime).getTime() + 60 * 60 * 1000 <= predictedAt
    : false;

  // Built from results.json so the wording follows any retrain.
  const horizonNote = useMemo(() => {
    if (!selection) return "";
    const losing = selection.horizons_not_beating_persistence ?? [];
    const shown = `only +${selection.horizon_min} min is shown`;
    if (!losing.length) return `+${selection.horizon_min} min has the largest margin over persistence, so ${shown}.`;
    const list = losing.map((h) => `+${h}`).join(" and ");
    return `At ${list} min the model does not beat persistence on the main test, so ${shown}.`;
  }, [selection]);

  const weekendNote = useMemo(() => {
    if (!summary || !selection) return "";
    const main = summary.splits.find((s) => s.split === selection.primary_split);
    const days = main?.test_weekend_days ?? [];
    if (!days.length) return "The main test contains no weekend day.";
    const high = summary.short_term.results.find(
      (r) =>
        r.split === selection.primary_split &&
        r.horizon_min === selection.horizon_min &&
        r.subset === "weekend" &&
        r.method === "random_forest",
    )?.support_High;
    const dayText = days.length === 1 ? `a single weekend day (${days[0]})` : `${days.length} weekend days (${days.join(", ")})`;
    const highText = high == null ? "" : ` with ${high} High reading${high === 1 ? "" : "s"} at +${selection.horizon_min} min`;
    const noisy = high != null && high < 30 ? ", so its numbers are especially noisy" : "";
    return `The weekend part of the main test is ${dayText}${highText}${noisy}.`;
  }, [summary, selection]);

  return (
    <>
      <section className="workspace prototype-workspace">
        <aside className="control-card">
          <div className="section-kicker">TRAFFIC SIMULATION</div>
          <h2>Choose a road and time</h2>
          <p className="muted">
            Pick one of Masar&apos;s {points.length} TomTom measurement points on {roads.length} roads, or click a dot on
            the map.
          </p>

          <CitySelect city={city} onChange={onCityChange} />

          <label className="field-label">Road and point</label>
          <select
            className="field-input"
            suppressHydrationWarning
            value={selectedId ?? ""}
            onChange={(event) => {
              const point = points.find((item) => item.point_id === event.target.value);
              if (point) selectPoint(point);
            }}
          >
            {roads.map(([road, roadPoints]) => (
              <optgroup key={road} label={road}>
                {roadPoints.map((point) => (
                  <option key={point.point_id} value={point.point_id}>
                    {point.short_label}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>

          <div className="two-fields">
            <div>
              <label className="field-label">Date</label>
              <input
                type="date"
                lang="en-GB"
                suppressHydrationWarning
                value={date}
                onChange={(event) => {
                  setDate(event.target.value);
                  setSimulation(null);
                }}
                className="field-input"
              />
            </div>
            <div>
              <label className="field-label">Hour</label>
              <select
                suppressHydrationWarning
                value={hour}
                onChange={(event) => {
                  setHour(Number(event.target.value));
                  setSimulation(null);
                }}
                className="field-input"
              >
                {Array.from({ length: 24 }, (_, value) => (
                  <option key={value} value={value}>{formatHour(value)}</option>
                ))}
              </select>
            </div>
          </div>
          {weekendSelected && (
            <div className="weekend-note">
              Weekend prediction (Fri/Sat). Weekend traffic in Jeddah is much lighter than on Sun–Thu.
            </div>
          )}

          <button
            suppressHydrationWarning
            className="simulate-button"
            onClick={runSimulation}
            disabled={!selectedId || !date || simulating}
          >
            {simulating ? "Running…" : "Predict traffic"}
          </button>

          {selected && (
            <div className="selected-road-card">
              <div className="selected-road-top">
                <span>Selected point · {selected.frc}</span>
                <span className={levelClass(selected.weekday_peak_level)}>{selected.weekday_peak_level} at weekday peak</span>
              </div>
              <strong>{selected.label}</strong>
              <div className="selected-road-stats">
                <span>{Math.round(selected.weekday_avg_speed_kmh)} km/h weekday average</span>
                <span>{Math.round(selected.free_flow_speed_kmh)} km/h free flow</span>
              </div>
              <div className="safety-mini-row">
                <span>Usually slowest on weekdays around</span>
                <strong>{formatHour(selected.weekday_peak_hour)}</strong>
              </div>
            </div>
          )}

          <div className="control-footnote">
            {data
              ? `${trainedOn} (${readingsText}). Last reading used: ${formatDateTime(data.trained_through)}. Pilot data — not live traffic.`
              : "Pilot data — not live traffic."}
          </div>
        </aside>

        <div className="map-card">
          <div className="map-toolbar">
            <div>
              <div className="section-kicker">JEDDAH · MASAR PILOT DATA</div>
              <h2>Predicted level · {date ? formatDateHour(`${date}T${formatHour(hour)}:00`) : formatHour(hour)}</h2>
            </div>
            <div className="map-legend">
              <span><i className="dot-low" />Low</span>
              <span><i className="dot-medium" />Medium</span>
              <span><i className="dot-high" />High congestion</span>
            </div>
          </div>
          <JeddahMap points={points} levels={levels} selectedPointId={selectedId} onSelect={selectPoint} />
        </div>
      </section>

      {error && <div className="error-banner">{error}</div>}

      <section className="results-section simple-results" id="simulation-output">
        <div className="results-header">
          <div>
            <div className="section-kicker">SIMULATION OUTPUT</div>
            <h2>{simulation ? simulation.point.label : "Your result will appear here"}</h2>
          </div>
          {simulation && (
            <div className="result-time">
              {simulation.day_type === "weekend" && <span className="weekend-badge">Weekend prediction</span>}
              {formatDateHour(simulation.datetime)}
            </div>
          )}
        </div>

        {!simulation ? (
          <div className="empty-state compact-empty">
            <div>
              <strong>Select a point, date and hour.</strong>
              <span>Then press “Predict traffic” to see the prediction.</span>
            </div>
          </div>
        ) : (
          <>
            <div className={`result-grid ${nearLatest ? "result-grid-five" : "result-grid-four"}`}>
              <div className="result-card">
                <span>Expected speed</span>
                <strong>{Math.round(simulation.predicted_speed_kmh)} km/h</strong>
                <small>free flow {Math.round(simulation.free_flow_speed_kmh)} km/h</small>
              </div>
              <div className="result-card">
                <span>Congestion level</span>
                <strong className={`result-level level-text-${simulation.traffic_level.toLowerCase()}`}>
                  {simulation.traffic_level}
                </strong>
                <small>
                  {simulation.historical_range.lower_speed_kmh != null && simulation.historical_range.upper_speed_kmh != null
                    ? `${Math.round(simulation.historical_range.lower_speed_kmh)}–${Math.round(simulation.historical_range.upper_speed_kmh)} km/h on past ${simulation.day_type}s at this hour`
                    : `No past ${simulation.day_type} readings at this hour`}
                </small>
              </div>
              <div className="result-card">
                <span>Speed vs free flow</span>
                <strong>{ratioPercent(simulation.predicted_speed_ratio)}</strong>
                <small>Low ≥ 85% · High &lt; 60%</small>
              </div>
              <div className="result-card">
                <span>Expected delay on this segment</span>
                {(() => {
                  const delay = delayText(simulation.expected_travel_time_s, simulation.free_flow_travel_time_s);
                  return (
                    <>
                      <strong>{delay.delta}</strong>
                      <small>{delay.detail}</small>
                    </>
                  );
                })()}
              </div>
              {nearLatest && (
                <NextHourCard forecast={simulation.short_term} horizon={simulation.short_term_horizon_min} />
              )}
            </div>
            <p className="date-note">
              This prediction depends only on the day type ({simulation.day_type}) and the hour, not on the specific
              date.
              {selectedIsPast && " This is a prediction, not the observed value."}
            </p>

            <div className="chart-card simple-chart-card">
              <div className="chart-header">
                <div>
                  <h3>12-hour speed forecast</h3>
                  <p>Expected average speed from the selected hour onward.</p>
                </div>
                <span className="model-chip">{simulation.method.label}</span>
              </div>
              <ForecastChart
                data={simulation.forecast.map((point) => ({
                  datetime: point.datetime,
                  hour: point.hour,
                  traffic: Math.round(point.speed_kmh),
                  level: point.level,
                }))}
                unit="km/h"
                seriesName="Expected speed"
              />
              <p className="confidence-note">
                <strong>Data coverage: {simulation.data_coverage.summary}.</strong>
                {simulation.data_coverage.note && ` ${simulation.data_coverage.note}`}
              </p>
            </div>
          </>
        )}
      </section>

      {shortTerm && (
        <section className="results-section" id="next-hour">
          <div className="results-header">
            <div>
              <div className="section-kicker">SHORT-TERM FORECAST</div>
              <h2>{horizonTitle(shortTerm.horizon_min)} (from latest reading)</h2>
              <p className="results-sub">
                Latest reading: {formatDateTime(latestReading ?? shortTerm.latest_timestamp)}
                {stepMin ? ` (${formatClock(shortTerm.latest_timestamp)} slot on the ${stepMin}-minute grid)` : ""}. Forecast
                for{" "}
                {formatDateTime(shortTerm.points[0]?.forecast_for ?? shortTerm.latest_timestamp)} (+{horizon} min), by the
                Random Forest. This does not depend on the date and hour chosen above.
              </p>
            </div>
            <span className="model-tag model-tag-final">Full-data model</span>
          </div>
          <div className="next-hour-grid">
            <div>
              <NextHourCard forecast={selectedNext} horizon={shortTerm.horizon_min} />
              {selected && <p className="next-hour-point">{selected.label}</p>}
            </div>
            <div className="chart-card model-card next-hour-table">
              <div className="nowcast-table-wrap">
                <table className="nowcast-table">
                  <thead>
                    <tr>
                      <th>Point</th>
                      <th>Latest (= persistence)</th>
                      <th>+{horizon} min model</th>
                      <th className="num">Model probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shortTerm.points.map((row) => {
                      const point = points.find((item) => item.point_id === row.point_id);
                      return (
                        <tr key={row.point_id} className={row.point_id === selectedId ? "row-selected" : ""}>
                          <td>{point?.short_label ?? row.point_id}</td>
                          <td><span className={levelClass(row.persistence_level)}>{row.persistence_level}</span></td>
                          <td><span className={levelClass(row.predicted_level)}>{row.predicted_level}</span></td>
                          <td className="num">{pct(row.model_probability)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <small>
                Model probability is the forest&apos;s average class probability, not a calibrated confidence.{" "}
                {horizonNote}
              </small>
            </div>
          </div>
        </section>
      )}

      {summary && selection && models && (
        <section className="results-section">
          <div className="results-header">
            <div>
              <div className="section-kicker">HOW THE PREDICTION WORKS</div>
              <h2>{trainedOn}</h2>
              <p className="results-sub">
                {readingsText} · {dayCount}. Chronological tests only: trained on
                earlier days, tested on later ones.
              </p>
            </div>
          </div>
          <div className="model-explain">
            <div>
              <span className="model-tag model-tag-eval">Evaluated model</span>
              <p>
                Trained on {formatDateRange(models.evaluated.train_start, models.evaluated.train_end)}, tested on{" "}
                {formatDateRange(models.evaluated.test_start, models.evaluated.test_end)}, which it never saw in training.
                Every score below comes from this held-out test.
              </p>
            </div>
            <div>
              <span className="model-tag model-tag-final">Full-data model (used for predictions)</span>
              <p>
                Same settings, refitted on all {models.final.rows.toLocaleString("en-GB")} readings (
                {formatDateRange(models.final.train_start, models.final.train_end)}) after the tests. It makes every
                prediction on this page and has no separate score of its own.
              </p>
            </div>
          </div>
          <div className="model-grid">
            <div className="chart-card model-card">
              <h3>Date &amp; hour → {summary.time_of_day.selected.label}</h3>
              <p>
                Used for the cards, the map colours and the 12-hour chart. Compared with historical averages on the same
                tests; the app uses the one with the lowest speed error on the main test (
                {splitLabel(models.evaluated.split)}).
              </p>
              <div className="nowcast-table-wrap">
                <table className="nowcast-table">
                  <thead>
                    <tr>
                      <th>Method</th>
                      <th className="num">Speed error</th>
                      <th className="num">Level acc.</th>
                      <th>Test set</th>
                    </tr>
                  </thead>
                  <tbody>
                    {timeRows.map((row) => (
                      <tr
                        key={`${row.split}-${row.subset}-${row.method}`}
                        className={[
                          row.method === summary.time_of_day.selected.method ? "row-selected" : "",
                          row.subset !== "all" ? "subset-row" : "",
                        ].join(" ")}
                      >
                        <td>{TIME_METHOD_SHORT[row.method] ?? summary.time_of_day.methods[row.method]}</td>
                        <td className="num">±{row.mae_speed_kmh.toFixed(1)} km/h</td>
                        <td className="num">{pct(row.accuracy)}</td>
                        <td>{testLabel(row.split, row.subset)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <small>
                Speed error = mean absolute error of the predicted speed. Level acc. = share of readings with the right
                Low / Medium / High level.
              </small>
            </div>

            <div className="chart-card model-card">
              <h3>Short term → Random Forest at +{horizon} min</h3>
              <p className="verdict">{selection.sentence}</p>
              <div className="nowcast-table-wrap">
                <table className="nowcast-table">
                  <thead>
                    <tr>
                      <th>Horizon</th>
                      <th className="num">Model F1</th>
                      <th className="num">Persist. F1</th>
                      <th className="num">Hist. F1</th>
                      <th className="num">High recall model / persist.</th>
                      <th>Test set</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shortRows.map((row) => (
                      <tr
                        key={`${row.split}-${row.horizon}-${row.subset}`}
                        className={[
                          row.horizon === horizon ? "row-selected" : "",
                          row.subset !== "all" ? "subset-row" : "",
                        ].join(" ")}
                      >
                        <td>+{row.horizon} min</td>
                        <td className="num">{row.rf?.macro_f1.toFixed(3)}</td>
                        <td className="num">{row.persistence?.macro_f1.toFixed(3)}</td>
                        <td className="num">{row.historical?.macro_f1.toFixed(3)}</td>
                        <td className="num">
                          {pct(row.rf?.recall_High)} / {pct(row.persistence?.recall_High)}
                          <span className="support"> (n={row.rf?.support_High})</span>
                        </td>
                        <td>{testLabel(row.split, row.subset)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <small>
                F1 = macro-F1 over Low / Medium / High. n = High readings in that test. {weekendNote} All results are
                pilot-only and indicative.
              </small>
            </div>
          </div>
        </section>
      )}

      {loading && <div className="loading-overlay">Loading Jeddah data…</div>}
    </>
  );
}
