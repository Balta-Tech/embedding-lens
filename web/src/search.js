// Search: text to video. Presets are precomputed; free text needs the local backend.
import { backend, formatTime, liveSearch, loadSearch, spriteUrl, timeOf, youtubeUrl } from "./data.js";
import { offsetOf } from "./explorer.js";

const SHOW = 8;

export function createSearch(manifest, state, explorer) {
  const $ = (id) => document.getElementById(id);
  const form = $("search-form");
  const input = $("search-input");
  const presets = $("search-presets");
  const hint = $("search-hint");
  const results = $("search-results");
  const lanes = $("search-lanes");
  let active = null;

  async function renderPresets() {
    const data = await loadSearch(state.model);
    presets.innerHTML = "";
    for (const q of data.queries) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = q;
      b.setAttribute("aria-pressed", q === active);
      b.addEventListener("click", () => run(q));
      presets.append(b);
    }
  }

  function updateHint() {
    hint.textContent = backend.live
      ? "Live: any sentence works, embedded on your machine."
      : "Offline: try the suggestions. Free text needs the local backend (see the status pill above).";
  }

  async function run(query) {
    active = query;
    input.value = query;
    const data = await loadSearch(state.model);
    let scores = data.scores[query];
    if (!scores) {
      if (!backend.live) {
        hint.textContent = "That query isn't precomputed. Start the local backend to search freely.";
        return;
      }
      hint.textContent = "Searching…";
      const live = await liveSearch(query, state.model);
      scores = live.segments.map((s) => s.z);
      updateHint();
    }
    renderPresets();
    render(scores);
  }

  function keyAt(i) {
    let rest = i;
    for (const v of manifest.videos) {
      if (rest < v.segments) return `${v.id}:${rest}`;
      rest -= v.segments;
    }
    return null;
  }

  function render(scores) {
    const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]).slice(0, SHOW);
    results.innerHTML = "";
    order.forEach(([score, i], n) => {
      const key = keyAt(i);
      const video = manifest.videos.find((v) => v.id === key.split(":")[0]);
      const a = document.createElement("a");
      a.className = "result";
      a.href = youtubeUrl(manifest, key);
      a.target = "_blank";
      a.rel = "noopener";
      a.innerHTML = `
        <div class="sprite" style="background-image:url(${spriteUrl(key)})"></div>
        <span class="rank">#${n + 1}</span>
        <div class="meta"><span>${video.title.split(",")[0]} · ${formatTime(timeOf(manifest, key))}</span><span>${score.toFixed(1)}σ</span></div>`;
      results.append(a);
    });

    // Where in each video the query lights up
    lanes.innerHTML = "";
    const max = Math.max(...scores);
    for (const v of manifest.videos) {
      const start = offsetOf(manifest, v.id);
      const row = document.createElement("div");
      row.className = "search-lane";
      row.innerHTML = `<span>${v.title.split(",")[0]}</span><div class="cells"></div>`;
      const cells = row.querySelector(".cells");
      for (let j = 0; j < v.segments; j++) {
        const s = scores[start + j];
        const cell = document.createElement("span");
        cell.style.opacity = Math.max(0.05, Math.min(1, s / max)).toFixed(2);
        cell.title = `${formatTime(j * manifest.segmentSeconds)}  ${s.toFixed(2)}σ`;
        cell.addEventListener("click", () => {
          explorer.set({ video: v.id, index: j });
          document.getElementById("explore").scrollIntoView({ behavior: "smooth" });
        });
        cells.append(cell);
      }
      lanes.append(row);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (q) run(q);
  });
  backend.listeners.add(updateHint);
  updateHint();
  renderPresets().then(() => run("mountains"));
  return { run, refresh: () => active && run(active) };
}
