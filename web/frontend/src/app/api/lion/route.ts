import { NextRequest, NextResponse } from "next/server";

const LION_QUERY =
  "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/LION/FeatureServer/0/query";

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as { segmentIds?: string[] };
    const ids = (body.segmentIds ?? [])
      .map((id) => String(id).replace(/[^0-9-]/g, ""))
      .filter(Boolean)
      .slice(0, 180);

    if (!ids.length) {
      return NextResponse.json({ type: "FeatureCollection", features: [] });
    }

    const where = `SegmentID IN (${ids.map((id) => `'${id}'`).join(",")})`;
    const params = new URLSearchParams({
      where,
      outFields: "SegmentID,Street,TrafDir,POSTED_SPEED,Number_Travel_Lanes",
      returnGeometry: "true",
      outSR: "4326",
      f: "geojson",
    });

    const response = await fetch(`${LION_QUERY}?${params.toString()}`, {
      next: { revalidate: 60 * 60 * 24 },
    });

    if (!response.ok) {
      return NextResponse.json(
        { type: "FeatureCollection", features: [], error: "LION geometry service unavailable" },
        { status: 502 },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { type: "FeatureCollection", features: [], error: "Could not load road geometry" },
      { status: 500 },
    );
  }
}
