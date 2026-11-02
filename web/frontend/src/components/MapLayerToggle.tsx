"use client";

import styles from "./safety.module.css";

export type MapLayer = "congestion" | "safety";

/** Congestion / Safety risk switch for the Jeddah map, with the matching legend. */
export default function MapLayerToggle({
  layer,
  onChange,
}: {
  layer: MapLayer;
  onChange: (layer: MapLayer) => void;
}) {
  return (
    <div className={styles.toggleWrap}>
      <div className={styles.toggle} role="group" aria-label="Map layer">
        {(["congestion", "safety"] as const).map((value) => (
          <button
            key={value}
            type="button"
            className={layer === value ? styles.active : ""}
            aria-pressed={layer === value}
            onClick={() => onChange(value)}
            suppressHydrationWarning
          >
            {value === "congestion" ? "Congestion" : "Safety risk"}
          </button>
        ))}
      </div>
      <div className="map-legend">
        <span><i className="dot-low" />Low</span>
        <span><i className="dot-medium" />Medium</span>
        <span><i className="dot-high" />{layer === "congestion" ? "High congestion" : "High risk"}</span>
      </div>
    </div>
  );
}
