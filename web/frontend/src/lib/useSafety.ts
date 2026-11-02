"use client";

import { useEffect, useMemo, useState } from "react";
import { SHOW_SAFETY } from "./features";
import { safetyApi, type SafetyLatest, type SafetyLevel } from "./safety";

/** Safety risk for every point at the chosen date and hour (usual pattern) and each point's latest reading. */
export function useSafety(date: string, hour: number) {
  const [levels, setLevels] = useState<Record<string, SafetyLevel>>({});
  const [latest, setLatest] = useState<Record<string, SafetyLatest>>({});

  useEffect(() => {
    if (!SHOW_SAFETY) return;
    safetyApi
      .latest()
      .then((rows) => setLatest(Object.fromEntries(rows.map((row) => [row.point_id, row]))))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!SHOW_SAFETY || !date) return;
    let cancelled = false;
    safetyApi
      .levels(date, hour)
      .then((rows) => {
        if (!cancelled) setLevels(Object.fromEntries(rows.map((row) => [row.point_id, row])));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [date, hour]);

  // Colour + popup text for the map's safety layer.
  const overrides = useMemo(
    () =>
      Object.fromEntries(
        Object.values(levels).map((row) => [
          row.point_id,
          {
            level: row.band,
            text: `Safety risk ${row.band} · ${Math.round(row.score)}/100${
              row.main_component !== "none" ? ` · ${row.main_component}` : ""
            }`,
          },
        ]),
      ),
    [levels],
  );

  return { levels, latest, overrides };
}
