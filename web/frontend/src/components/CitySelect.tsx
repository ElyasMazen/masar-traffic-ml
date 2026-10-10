"use client";

export type City = "jeddah" | "new-york";

export default function CitySelect({
  city,
  onChange,
}: {
  city: City;
  onChange: (city: City) => void;
}) {
  return (
    <>
      <label className="field-label">City</label>
      <select
        className="field-input"
        value={city}
        onChange={(event) => onChange(event.target.value as City)}
      >
        <option value="jeddah">Jeddah — Masar pilot</option>
        <option value="new-york">New York — reference data</option>
      </select>
    </>
  );
}
