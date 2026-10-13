"use client";

export type City = "jeddah" | "new-york";

// The demo is Jeddah only. The NYC reference view still works; set this to
// true to offer it in the dropdown again.
const SHOW_NYC = false;

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
        suppressHydrationWarning
      >
        <option value="jeddah">Jeddah — Masar pilot</option>
        {SHOW_NYC && <option value="new-york">New York — reference data</option>}
      </select>
    </>
  );
}
