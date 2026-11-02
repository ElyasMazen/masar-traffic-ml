"use client";

import { formatDateTime, formatHour, levelClass } from "@/lib/format";
import type { SafetyLatest, SafetyLevel } from "@/lib/safety";
import styles from "./safety.module.css";

export const SAFETY_NOTE =
  "Indicator based on speed patterns linked to crash risk in the literature; not validated against crash data.";

/** "Safety risk" card for the selected point, date and hour (usual pattern), plus its latest reading. */
export default function SafetyCard({
  level,
  latest,
  note = SAFETY_NOTE,
}: {
  level: SafetyLevel | null;
  latest: SafetyLatest | null;
  note?: string;
}) {
  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <span className={styles.title}>Safety risk</span>
        {level && <span className={levelClass(level.band)}>{level.band}</span>}
      </div>
      {level ? (
        <>
          <div className={styles.scoreRow}>
            <strong className={`level-text-${level.band.toLowerCase()}`}>{Math.round(level.score)}</strong>
            <span>/ 100</span>
          </div>
          <p className={styles.reason}>{level.reason}.</p>
          <small className={styles.meta}>
            Usual pattern for {level.day_type}s at {formatHour(level.hour)}: the score is reached on the worse days
            (90th percentile of {level.readings} readings, {level.days} days). Medium or High in{" "}
            {Math.round(level.share_elevated * 100)}% of those readings.
          </small>
        </>
      ) : (
        <p className={styles.reason}>No safety data for this point and hour.</p>
      )}
      {latest && (
        <small className={styles.meta}>
          Latest reading ({formatDateTime(latest.reading_time)}): {Math.round(latest.score)} / 100 · {latest.band}
          {latest.main_component !== "none" ? ` · ${latest.main_component}` : ""}
        </small>
      )}
      <p className={styles.note}>{note}</p>
    </div>
  );
}
