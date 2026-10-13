// MapLibre 6 looks for its worker next to its own module (import.meta.url).
// Once Turbopack bundles MapLibre into a chunk, that file is not there, so the
// map logs "Worker failed to load". Serve the worker from public/ instead;
// src/lib/maplibre.ts points MapLibre at it. Runs before dev and build.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = createRequire(import.meta.url).resolve("maplibre-gl/package.json");
const source = join(dirname(pkg), "dist", "maplibre-gl-worker.mjs");

mkdirSync(join(root, "public"), { recursive: true });
copyFileSync(source, join(root, "public", "maplibre-gl-worker.mjs"));
