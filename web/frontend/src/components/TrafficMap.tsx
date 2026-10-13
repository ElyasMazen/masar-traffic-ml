"use client";

import { useEffect, useRef, useState } from "react";
import {
  Map as MapLibreMap,
  NavigationControl,
  type GeoJSONSource,
  type StyleSpecification,
} from "@/lib/maplibre";

import type { MapRoad, PoliceRoadState, SegmentRoad } from "@/lib/api";
import { LOCALE } from "@/lib/format";

type Props = {
  roads: MapRoad[];
  segments: SegmentRoad[];
  selectedRoadKey: string | null;
  onSelect: (road: MapRoad) => void;
  policeRoadStates?: Record<string, PoliceRoadState>;
  policeSimulationActive?: boolean;
};


const NYC_BOUNDS: [[number, number], [number, number]] = [
  [-74.30, 40.45],
  [-73.65, 40.95],
];

export const baseStyle: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

type GeoFeature = {
  type: "Feature";
  geometry: unknown;
  properties: Record<string, unknown>;
};

type GeoCollection = {
  type: "FeatureCollection";
  features: GeoFeature[];
};

function roadPoints(roads: MapRoad[], selectedRoadKey: string | null): GeoCollection {
  return {
    type: "FeatureCollection",
    features: roads
      .filter((road) => road.latitude != null && road.longitude != null)
      .map((road) => ({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [road.longitude!, road.latitude!],
        },
        properties: {
          road_key: road.road_key,
          display_name: road.display_name,
          risk_score: road.risk_score,
          risk_level: road.risk_level,
          selected: road.road_key === selectedRoadKey,
        },
      })),
  };
}

function chunk<T>(items: T[], size: number) {
  const out: T[][] = [];
  for (let index = 0; index < items.length; index += size) {
    out.push(items.slice(index, index + size));
  }
  return out;
}

async function fetchLionGeometry(segments: SegmentRoad[]): Promise<GeoCollection> {
  const meta = new globalThis.Map(segments.map((segment) => [String(segment.segment_id), segment]));
  const batches = chunk(segments.map((segment) => String(segment.segment_id)), 140);

  const responses = await Promise.all(
    batches.map(async (ids) => {
      const response = await fetch("/api/lion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segmentIds: ids }),
      });
      if (!response.ok) return [] as GeoFeature[];
      const data = (await response.json()) as { features?: GeoFeature[] };
      return data.features ?? [];
    }),
  );

  const features = responses.flat().flatMap((feature) => {
    const segmentId = String(feature.properties?.SegmentID ?? "");
    const segment = meta.get(segmentId);
    if (!segment) return [];
    return [
      {
        ...feature,
        properties: {
          ...feature.properties,
          segment_id: segment.segment_id,
          road_key: segment.road_key,
          display_name: segment.display_name,
          from_street: segment.from_street ?? "",
          to_street: segment.to_street ?? "",
          direction: segment.direction ?? "",
          risk_score: segment.risk_score,
          safety_score: segment.safety_score,
          risk_level: segment.risk_level,
          historical_crashes: segment.historical_crashes,
          avg_traffic: segment.avg_traffic,
          latitude: segment.latitude,
          longitude: segment.longitude,
        },
      },
    ];
  });

  return { type: "FeatureCollection", features };
}

function selectGeometry(
  collection: GeoCollection,
  selectedRoadKey: string | null,
  policeRoadStates: Record<string, PoliceRoadState> = {},
  policeSimulationActive = false,
): GeoCollection {
  return {
    type: "FeatureCollection",
    features: collection.features.map((feature) => {
      const roadKey = String(feature.properties.road_key ?? "");
      const policeState = policeRoadStates[roadKey];
      return {
        ...feature,
        properties: {
          ...feature.properties,
          selected: feature.properties.road_key === selectedRoadKey,
          police_active: policeSimulationActive,
          sim_traffic_level: policeState?.traffic_level ?? "Low",
          sim_predicted_traffic: policeState?.predicted_traffic ?? 0,
          sim_alert_level: policeState?.alert_level ?? "Normal",
        },
      };
    }),
  };
}

export default function TrafficMap({
  roads,
  segments,
  selectedRoadKey,
  onSelect,
  policeRoadStates = {},
  policeSimulationActive = false,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const geometryRef = useRef<GeoCollection>({ type: "FeatureCollection", features: [] });
  const roadsRef = useRef<MapRoad[]>(roads);
  const segmentsRef = useRef<SegmentRoad[]>(segments);
  const selectRef = useRef(onSelect);
  const selectedRef = useRef<string | null>(selectedRoadKey);
  const [geometryStatus, setGeometryStatus] = useState("Loading official NYC road segments…");

  useEffect(() => {
    roadsRef.current = roads;
  }, [roads]);

  useEffect(() => {
    segmentsRef.current = segments;
  }, [segments]);

  useEffect(() => {
    selectRef.current = onSelect;
  }, [onSelect]);

  useEffect(() => {
    selectedRef.current = selectedRoadKey;
  }, [selectedRoadKey]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new MapLibreMap({
      container: containerRef.current,
      style: baseStyle,
      center: [-73.9855, 40.7484],
      zoom: 10.65,
      minZoom: 10.35,
      maxZoom: 18,
      maxBounds: NYC_BOUNDS,
      renderWorldCopies: false,
      pitch: 0,
      attributionControl: { compact: true },
    });

    map.addControl(new NavigationControl({ showCompass: false }), "top-right");

    map.on("load", () => {
      map.addSource("risk-points", {
        type: "geojson",
        data: roadPoints(roadsRef.current, selectedRef.current) as never,
      });

      map.addLayer({
        id: "risk-heat",
        type: "heatmap",
        source: "risk-points",
        maxzoom: 13,
        paint: {
          "heatmap-weight": [
            "interpolate",
            ["linear"],
            ["get", "risk_score"],
            0,
            0,
            100,
            1,
          ],
          "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 9, 0.35, 13, 0.9],
          "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 9, 18, 13, 34],
          "heatmap-opacity": ["interpolate", ["linear"], ["zoom"], 9, 0.22, 13, 0.04],
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0,
            "rgba(61,95,255,0)",
            0.4,
            "rgba(69,101,239,0.18)",
            0.72,
            "rgba(224,157,54,0.28)",
            1,
            "rgba(220,90,85,0.42)",
          ],
        },
      });

      map.addSource("traffic-segments", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });

      map.addLayer({
        id: "segment-outline",
        type: "line",
        source: "traffic-segments",
        paint: {
          "line-color": "#ffffff",
          "line-width": ["case", ["==", ["get", "selected"], true], 9, 6],
          "line-opacity": 0.96,
        },
      });

      map.addLayer({
        id: "traffic-segment-lines",
        type: "line",
        source: "traffic-segments",
        paint: {
          "line-color": [
            "case",
            ["==", ["get", "police_active"], true],
            [
              "match",
              ["get", "sim_traffic_level"],
              "High",
              "#d64f49",
              "Medium",
              "#d6a13b",
              "#269b72",
            ],
            ["==", ["get", "selected"], true],
            "#334ed8",
            [
              "match",
              ["get", "risk_level"],
              "High",
              "#dc5a55",
              "Medium",
              "#d99c2b",
              "#269b72",
            ],
          ],
          "line-width": [
            "interpolate",
            ["linear"],
            ["zoom"],
            9,
            ["case", ["==", ["get", "selected"], true], 5.8, 2.2],
            14,
            ["case", ["==", ["get", "selected"], true], 9, 5.4],
          ],
          "line-opacity": ["case", ["==", ["get", "selected"], true], 1, 0.88],
        },
      });

      map.on("mouseenter", "traffic-segment-lines", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "traffic-segment-lines", () => {
        map.getCanvas().style.cursor = "";
      });

      map.on("click", "traffic-segment-lines", (event) => {
        const feature = event.features?.[0];
        const props = feature?.properties;
        const roadKey = props?.road_key as string | undefined;
        if (!roadKey) return;

        const existing = roadsRef.current.find((road) => road.road_key === roadKey);
        if (existing) {
          selectRef.current(existing);
          return;
        }

        const segment = segmentsRef.current.find((item) => item.road_key === roadKey);
        if (!segment) return;
        selectRef.current({
          road_key: segment.road_key,
          display_name: segment.display_name,
          historical_crashes: segment.historical_crashes,
          risk_level: segment.risk_level,
          risk_score: segment.risk_score,
          safety_score: segment.safety_score,
          avg_traffic: segment.avg_traffic,
          latitude: segment.latitude,
          longitude: segment.longitude,
        });
      });
    });

    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!segments.length) return;
    let cancelled = false;
    setGeometryStatus("Loading official NYC road segments…");

    fetchLionGeometry(segments)
      .then((geometry) => {
        if (cancelled) return;
        geometryRef.current = geometry;
        const source = mapRef.current?.getSource("traffic-segments") as GeoJSONSource | undefined;
        source?.setData(selectGeometry(geometry, selectedRef.current) as never);
        setGeometryStatus(
          geometry.features.length > 0
            ? `${geometry.features.length.toLocaleString(LOCALE)} clickable road segments loaded`
            : "Road geometry unavailable — use road search",
        );
      })
      .catch(() => {
        if (!cancelled) setGeometryStatus("Road geometry unavailable — use road search");
      });

    return () => {
      cancelled = true;
    };
  }, [segments]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const points = map.getSource("risk-points") as GeoJSONSource | undefined;
    points?.setData(roadPoints(roads, selectedRoadKey) as never);

    const lines = map.getSource("traffic-segments") as GeoJSONSource | undefined;
    if (geometryRef.current.features.length) {
      lines?.setData(
        selectGeometry(
          geometryRef.current,
          selectedRoadKey,
          policeRoadStates,
          policeSimulationActive,
        ) as never,
      );
    }
  }, [roads, selectedRoadKey, policeRoadStates, policeSimulationActive]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !selectedRoadKey) return;
    const road = roads.find((item) => item.road_key === selectedRoadKey);
    if (road?.longitude == null || road?.latitude == null) return;
    map.flyTo({
      center: [road.longitude, road.latitude],
      zoom: Math.max(map.getZoom(), 12.1),
      duration: 720,
    });
  }, [selectedRoadKey, roads]);

  return (
    <div className="traffic-map-wrap">
      <div ref={containerRef} className="traffic-map" />
      <div className="map-data-status">
        <span className="map-data-dot" />
        {geometryStatus}
      </div>
    </div>
  );
}
