"use client";

import { useEffect, useRef } from "react";
import {
  Map as MapLibreMap,
  NavigationControl,
  Popup,
  type GeoJSONSource,
} from "maplibre-gl";

import { baseStyle } from "./TrafficMap";
import type { JeddahLevel, JeddahPoint } from "@/lib/jeddah";

type Props = {
  points: JeddahPoint[];
  levels: Record<string, JeddahLevel>;
  selectedPointId: string | null;
  onSelect: (point: JeddahPoint) => void;
};

const JEDDAH_BOUNDS: [[number, number], [number, number]] = [
  [38.9, 21.3],
  [39.5, 21.95],
];

function pointCollection(
  points: JeddahPoint[],
  levels: Record<string, JeddahLevel>,
  selectedPointId: string | null,
) {
  return {
    type: "FeatureCollection" as const,
    features: points.map((point) => {
      const level = levels[point.point_id];
      return {
        type: "Feature" as const,
        geometry: { type: "Point" as const, coordinates: [point.longitude, point.latitude] },
        properties: {
          point_id: point.point_id,
          label: point.label,
          traffic_level: level?.traffic_level ?? "Unknown",
          speed_kmh: level?.speed_kmh ?? null,
          selected: point.point_id === selectedPointId,
        },
      };
    }),
  };
}

export default function JeddahMap({ points, levels, selectedPointId, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const pointsRef = useRef(points);
  const dataRef = useRef(pointCollection(points, levels, selectedPointId));
  const selectRef = useRef(onSelect);

  useEffect(() => {
    pointsRef.current = points;
  }, [points]);

  useEffect(() => {
    selectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new MapLibreMap({
      container: containerRef.current,
      style: baseStyle,
      center: [39.165, 21.635],
      zoom: 10.6,
      minZoom: 9.5,
      maxZoom: 18,
      maxBounds: JEDDAH_BOUNDS,
      renderWorldCopies: false,
      attributionControl: { compact: true },
    });
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");

    const popup = new Popup({ closeButton: false, closeOnClick: false, offset: 14 });

    map.on("load", () => {
      map.addSource("jeddah-points", {
        type: "geojson",
        data: dataRef.current as never,
      });

      map.addLayer({
        id: "jeddah-point-halo",
        type: "circle",
        source: "jeddah-points",
        paint: {
          "circle-radius": ["case", ["==", ["get", "selected"], true], 22, 15],
          "circle-color": "#ffffff",
          "circle-opacity": 0.92,
          "circle-stroke-width": ["case", ["==", ["get", "selected"], true], 3, 0],
          "circle-stroke-color": "#334ed8",
        },
      });

      map.addLayer({
        id: "jeddah-point-dots",
        type: "circle",
        source: "jeddah-points",
        paint: {
          "circle-radius": ["case", ["==", ["get", "selected"], true], 14, 10],
          "circle-color": [
            "match",
            ["get", "traffic_level"],
            "High",
            "#d64f49",
            "Medium",
            "#d6a13b",
            "Low",
            "#269b72",
            "#9aa7a2",
          ],
        },
      });

      map.on("mouseenter", "jeddah-point-dots", (event) => {
        map.getCanvas().style.cursor = "pointer";
        const props = event.features?.[0]?.properties;
        if (!props) return;
        const speed = props.speed_kmh == null || props.speed_kmh === "null" ? "" : ` · ${Math.round(Number(props.speed_kmh))} km/h`;
        popup
          .setLngLat(event.lngLat)
          .setText(`${props.label} — ${props.traffic_level}${speed}`)
          .addTo(map);
      });
      map.on("mouseleave", "jeddah-point-dots", () => {
        map.getCanvas().style.cursor = "";
        popup.remove();
      });
      map.on("click", "jeddah-point-dots", (event) => {
        const id = event.features?.[0]?.properties?.point_id as string | undefined;
        const point = pointsRef.current.find((item) => item.point_id === id);
        if (point) selectRef.current(point);
      });
    });

    mapRef.current = map;
    return () => {
      popup.remove();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    // Before the map loads, the load handler picks this up from dataRef.
    dataRef.current = pointCollection(points, levels, selectedPointId);
    const source = mapRef.current?.getSource("jeddah-points") as GeoJSONSource | undefined;
    source?.setData(dataRef.current as never);
  }, [points, levels, selectedPointId]);

  useEffect(() => {
    const map = mapRef.current;
    const point = points.find((item) => item.point_id === selectedPointId);
    if (!map || !point) return;
    map.flyTo({
      center: [point.longitude, point.latitude],
      zoom: Math.max(map.getZoom(), 11.2),
      duration: 720,
    });
  }, [selectedPointId, points]);

  return (
    <div className="traffic-map-wrap">
      <div ref={containerRef} className="traffic-map" />
      <div className="map-data-status">
        <span className="map-data-dot" />
        {points.length
          ? `${points.length} TomTom measurement points on 3 Jeddah roads`
          : "Loading Jeddah measurement points…"}
      </div>
    </div>
  );
}
