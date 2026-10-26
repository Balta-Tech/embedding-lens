import "./styles.css";
import { drawBubbles } from "./bubbles.js";
import { backend, detectBackend, formatTime, loadManifest, loadReadouts, spriteUrl, timeOf } from "./data.js";
import { createExplorer } from "./explorer.js";
import { createGame } from "./game.js";
import { createLab } from "./lab.js";
import { createSearch } from "./search.js";
import { createTour } from "./tour.js";

// Moments the hero cycles through
const FEATURED = ["zugspitze:3", "zugspitze:50", "f16:18", "kart:20", "zugspitze:88", "trex:10"];

async function hero(manifest, state) {
  const sprite = document.getElementById("hero-sprite");
  const svg = document.getElementById("hero-bubbles");
  const caption = document.getElementById("hero-caption");
  const data = await loadReadouts(state.model);
  let i = 0;
  const show = () => {
    const key = FEATURED[i % FEATURED.length];
    const video = manifest.videos.find((v) => v.id === key.split(":")[0]);
    sprite.style.backgroundImage = `url(${spriteUrl(key)})`;
    caption.textContent = `${video.title.split(",")[0]} · ${formatTime(timeOf(manifest, key))}`;
    drawBubbles(svg, data.segments[key].calibrated, { count: 9, minFont: 11 });
    i++;
  };
  show();
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) setInterval(show, 4200);
}

function stats(manifest) {
  const segments = manifest.videos.reduce((n, v) => n + v.segments, 0);
  const set = (name, value) => document.querySelectorAll(`[data-stat="${name}"]`).forEach((el) => { el.textContent = value; });
  set("segments", segments.toLocaleString());
  set("seconds", manifest.segmentSeconds);
  set("vocab", manifest.vocab.size.toLocaleString());
}

function statusPill() {
  const pill = document.getElementById("status");
  const help = document.getElementById("backend-help");
  const update = (live) => {
    pill.classList.toggle("live", live);
    pill.querySelector(".status-text").textContent = live ? "Live backend" : "Precomputed";
  };
  backend.listeners.add(update);
  pill.addEventListener("click", () => { help.hidden = false; });
  document.getElementById("backend-close").addEventListener("click", () => { help.hidden = true; });
  help.addEventListener("click", (e) => { if (e.target === help) help.hidden = true; });
}

async function main() {
  const manifest = await loadManifest();
  const state = { model: manifest.defaultModel, lens: "calibrated", video: "zugspitze", index: 50 };
  stats(manifest);
  statusPill();
  hero(manifest, state);
  const explorer = createExplorer(manifest, state);
  await explorer.set({});
  const search = createSearch(manifest, state, explorer);
  const game = createGame(manifest, state);
  createLab(manifest);
  explorer.onChange(() => search.refresh());
  const tour = createTour(explorer, state);
  document.getElementById("start-tour").addEventListener("click", () => tour.start());
  detectBackend();
  void game;
}

main();
