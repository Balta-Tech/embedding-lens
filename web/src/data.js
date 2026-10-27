// Loading precomputed data, and talking to the optional live backend.

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
// Reranked read-outs exist only for some model and vocabulary pairs
export const loadRerank = (model, tier) => json(`readouts/${model}.${tier}.rerank.json`).catch(() => null);

export const loadGameVocab = (model) => json(`game/${model}/vocab.json`);

export async function loadGameRanks(model, key) {
  const [video, index] = key.split(":");
  const path = `game/${model}/${video}-${String(index).padStart(3, "0")}.bin`;
  if (!cache.has(path)) {
    cache.set(path, fetch(`/data/${path}`).then((r) => r.arrayBuffer()).then((b) => new Uint16Array(b)));
  }
  return cache.get(path);
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

// In development Vite proxies /api; the deployed site uses the hosted backend. Setting
// localStorage "embedding-lens:api" (e.g. to http://localhost:8000) points it elsewhere.
const API_HOST = "https://embedding-lens-api.balta-tech.com";

const API = (() => {
  try {
    const override = localStorage.getItem("embedding-lens:api");
    if (override) return override;
  } catch { /* storage unavailable */ }
  return location.hostname === "localhost" ? "" : API_HOST;
})();

export const backend = { live: false, listeners: new Set() };

export async function detectBackend() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
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
export const liveRerank = (segment, model, tier) => post("/api/rerank", { segment, model, tier });
