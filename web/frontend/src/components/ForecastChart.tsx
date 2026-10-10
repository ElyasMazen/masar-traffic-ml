"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ForecastPoint } from "@/lib/api";
import { LOCALE, formatHour } from "@/lib/format";

export default function ForecastChart({
  data,
  unit = "vehicles",
  seriesName = "Traffic",
}: {
  data: ForecastPoint[];
  unit?: string;
  seriesName?: string;
}) {
  const formatted = data.map((point) => ({
    ...point,
    label: formatHour(point.hour),
  }));

  return (
    <div className="chart-shell">
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={formatted} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id="trafficFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#4565ef" stopOpacity={0.22} />
              <stop offset="100%" stopColor="#4565ef" stopOpacity={0.015} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#edf0f4" vertical={false} />
          <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "#7c8494", fontSize: 12 }} />
          <YAxis axisLine={false} tickLine={false} tick={{ fill: "#7c8494", fontSize: 12 }} />
          <Tooltip
            cursor={{ stroke: "#cdd5e7" }}
            contentStyle={{
              borderRadius: 14,
              border: "1px solid #e5e8ef",
              boxShadow: "0 12px 34px rgba(22,31,52,.10)",
            }}
            formatter={(value) => [`${Number(value).toLocaleString(LOCALE)} ${unit}`, seriesName]}
          />
          <Area
            type="monotone"
            dataKey="traffic"
            stroke="#4565ef"
            strokeWidth={2.4}
            fill="url(#trafficFill)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
