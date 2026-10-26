// Loading precomputed data, and talking to the optional local backend.

const cache = new Map();

async function json(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(`/data/${path}`).then((r) => {
      if (!r.ok) throw new Error(`${path}: ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(path);
}

export const loadManifest = () => json("manifest.json");
export const loadReadouts = (model, tier) => json(`readouts/${model}.${tier}.json`);
export const loadLanes = (model) => json(`lanes/${model}.json`);
export const loadSearch = (model) => json(`search/${model}.json`);
export const loadResult = (name) => json(`results/${name}.json`).catch(() => null);

export async function loadGame(model) {
  const key = `game:${model}`;
  if (!cache.has(key)) {
    cache.set(key, Promise.all([
      json(`game/${model}.json`),
      fetch(`/data/game/${model}.ranks.bin`).then((r) => r.arrayBuffer()),
    ]).then(([meta, buffer]) => ({ ...meta, ranks: new Uint16Array(buffer) })));
  }
  return cache.get(key);
}

export function spriteUrl(key) {
  const [video, index] = key.split(":");
  return `/media/${video}/${String(index).padStart(3, "0")}.webp`;
}

export function timeOf(manifest, key) {
  const index = Number(key.split(":")[1]);
  return index * manifest.segmentSeconds;
}

export function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function youtubeUrl(manifest, key) {
  const video = manifest.videos.find((v) => v.id === key.split(":")[0]);
  const t = Math.floor(video.offset + timeOf(manifest, key));
  return `https://www.youtube.com/watch?v=${video.youtube}&t=${t}s`;
}

// --- live backend --------------------------------------------------------------------------

// In development Vite proxies /api; the deployed site looks for the backend on this machine.
const API = (() => {
  try {
    const override = localStorage.getItem("embedding-lens:api");
    if (override) return override;
  } catch { /* storage unavailable */ }
  return location.hostname === "localhost" ? "" : "http://localhost:8000";
})();

export const backend = { live: false, listeners: new Set() };

export async function detectBackend() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 1500);
  try {
    const r = await fetch(`${API}/api/health`, { signal: controller.signal });
    backend.live = r.ok;
  } catch {
    backend.live = false;
  } finally {
    clearTimeout(timer);
  }
  backend.listeners.forEach((fn) => fn(backend.live));
  return backend.live;
}

async function post(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

export const liveSearch = (query, model) => post("/api/search", { query, model });
export const liveScore = (segment, words, model) => post("/api/score", { segment, words, model });
