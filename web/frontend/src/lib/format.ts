// One locale for every date and time in the UI, regardless of the browser's language.
export const LOCALE = "en-GB";

export function formatHour(hour: number) {
  return `${String(hour).padStart(2, "0")}:00`;
}

/** Parses ISO or pandas "YYYY-MM-DD HH:MM:SS" (local Saudi time, no zone) as local time. */
export function toDate(value: string) {
  return new Date(value.replace(" ", "T"));
}

/** e.g. "Sat 10 Oct, 20:00" */
export function formatDateTime(iso: string) {
  return toDate(iso).toLocaleString(LOCALE, {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** e.g. "Sun 11 Oct, 17:00" for a selected date + hour */
export function formatDateHour(iso: string) {
  const d = toDate(iso);
  return `${d.toLocaleDateString(LOCALE, { weekday: "short", day: "numeric", month: "short" })}, ${formatHour(d.getHours())}`;
}

/** e.g. "5–10 Oct 2026" */
export function formatDateRange(startIso: string, endIso: string) {
  const start = toDate(startIso);
  const end = toDate(endIso);
  const sameMonth = start.getMonth() === end.getMonth() && start.getFullYear() === end.getFullYear();
  const startText = start.toLocaleDateString(LOCALE, sameMonth ? { day: "numeric" } : { day: "numeric", month: "short" });
  return `${startText}–${end.toLocaleDateString(LOCALE, { day: "numeric", month: "short", year: "numeric" })}`;
}

/** Local YYYY-MM-DD (toISOString would give the UTC date). */
export function isoDate(date: Date) {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function levelClass(level?: string) {
  return `level-pill level-${(level ?? "low").toLowerCase()}`;
}
