import { setWorkerUrl } from "maplibre-gl";

// Copied into public/ by scripts/copy-maplibre-worker.mjs (predev / prebuild).
// Without this MapLibre looks next to the bundled chunk and fails to load its worker.
setWorkerUrl("/maplibre-gl-worker.mjs");

export * from "maplibre-gl";
