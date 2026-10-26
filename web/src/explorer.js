// Explore: pick a video, scrub its moments, read each one out through a model and a lens.
import { drawBubbles } from "./bubbles.js";
import { formatTime, loadLanes, loadReadouts, spriteUrl, timeOf, youtubeUrl } from "./data.js";

const LENS_NOTES = {
  raw: "Plain cosine similarity. Honest, but words that sit close to every video tend to win.",
  calibrated:
    "The gap between video and text vectors removed, and words that are close to everything discounted (centring + CSLS). What is specific to this moment.",
  sparse:
    "A handful of words whose vectors add up to this moment's vector. Each one explains something the others don't.",
};
const LANE_COUNT = 6;

export function createExplorer(manifest, state) {
  const $ = (id) => document.getElementById(id);
  const tabs = $("video-tabs");
  const strip = $("strip");
  const bubbles = $("bubbles");
  const list = $("wordlist");
  const listeners = new Set();

  // Video tabs
  for (const video of manifest.videos) {
    const tab = document.createElement("button");
    tab.className = "video-tab";
    tab.type = "button";
    tab.role = "tab";
    tab.dataset.video = video.id;
    const middle = Math.floor(video.segments / 2);
    tab.innerHTML = `
      <span class="thumb" style="background-image:url(${spriteUrl(`${video.id}:${middle}`)})"></span>
      <span class="world">${video.world === "sim" ? "rendered" : "camera"}</span>
      <span class="name">${video.title}</span>`;
    tab.addEventListener("click", () => set({ video: video.id, index: Math.min(state.index, video.segments - 1) }));
    tabs.append(tab);
  }

  // Model and lens pickers
  const modelPicker = $("model-picker");
  for (const m of manifest.models) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.model = m.id;
    b.textContent = m.label;
    b.title = `${m.kind}, ${m.dim} dimensions`;
    b.addEventListener("click", () => set({ model: m.id }));
    modelPicker.append(b);
  }
  $("lens-picker").addEventListener("click", (e) => {
    const lens = e.target.closest("button")?.dataset.lens;
    if (lens) set({ lens });
  });

  strip.addEventListener("keydown", (e) => {
    const video = manifest.videos.find((v) => v.id === state.video);
    if (e.key === "ArrowRight") set({ index: Math.min(state.index + 1, video.segments - 1) });
    else if (e.key === "ArrowLeft") set({ index: Math.max(state.index - 1, 0) });
    else return;
    e.preventDefault();
  });

  let drawnStrip = null;

  async function renderStrip() {
    const video = manifest.videos.find((v) => v.id === state.video);
    const lanes = await loadLanes(state.model);
    const key = `${state.video}:${state.model}`;
    if (drawnStrip !== key) {
      drawnStrip = key;
      strip.innerHTML = "";
      const frames = row("frames", "");
      for (let i = 0; i < video.segments; i++) {
        const b = document.createElement("button");
        b.className = "frame-btn";
        b.type = "button";
        b.role = "option";
        b.dataset.index = i;
        b.title = formatTime(i * manifest.segmentSeconds);
        b.style.backgroundImage = `url(${spriteUrl(`${video.id}:${i}`)})`;
        b.addEventListener("click", () => set({ index: i }));
        frames.append(b);
      }
      strip.append(frames);
      // The concepts that vary most within this video are the interesting lanes
      const start = offsetOf(manifest, video.id);
      const ranked = lanes.concepts
        .map((c) => {
          const values = lanes.scores[c].slice(start, start + video.segments);
          return { c, values, spread: Math.max(...values) - Math.min(...values) };
        })
        .sort((a, b) => b.spread - a.spread)
        .slice(0, LANE_COUNT);
      for (const { c, values } of ranked) {
        const r = row("lane", c);
        for (const v of values) {
          const cell = document.createElement("span");
          cell.className = "lane-cell";
          cell.style.opacity = Math.max(0.04, Math.min(1, (v + 0.5) / 3)).toFixed(2);
          r.append(cell);
        }
        strip.append(r);
      }
    }
    strip.querySelectorAll(".frame-btn").forEach((b) => {
      const on = Number(b.dataset.index) === state.index;
      b.setAttribute("aria-selected", on);
      b.tabIndex = on ? 0 : -1;
    });
    const current = strip.querySelector(`.frame-btn[data-index="${state.index}"]`);
    if (current) {
      const target = current.offsetLeft - strip.clientWidth / 2 + current.clientWidth / 2;
      strip.scrollTo({ left: Math.max(0, target), behavior: "smooth" });
    }
  }

  function row(kind, label) {
    const r = document.createElement("div");
    r.className = `strip-row ${kind}`;
    const l = document.createElement("span");
    l.className = "strip-label";
    l.textContent = label;
    r.append(l);
    return r;
  }

  async function renderReadout() {
    const key = `${state.video}:${state.index}`;
    $("moment-sprite").style.backgroundImage = `url(${spriteUrl(key)})`;
    $("moment-time").textContent = formatTime(timeOf(manifest, key));
    $("moment-yt").href = youtubeUrl(manifest, key);

    tabs.querySelectorAll(".video-tab").forEach((t) => t.setAttribute("aria-selected", t.dataset.video === state.video));
    modelPicker.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.model === state.model));
    $("lens-picker").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.lens === state.lens));

    const data = (await loadReadouts(state.model)).segments[key];
    let words;
    let note = LENS_NOTES[state.lens];
    if (state.lens === "sparse") {
      words = data.sparse.terms;
      note += ` Together they reproduce the vector with cosine ${data.sparse.fit.toFixed(2)}.`;
    } else {
      words = data[state.lens];
    }
    $("lens-note").textContent = note;
    drawBubbles(bubbles, words, { count: state.lens === "sparse" ? 6 : 14 });
    list.innerHTML = words
      .slice(0, 12)
      .map(([p, s]) => `<li><span>${p}</span><span class="score">${s.toFixed(3)}</span></li>`)
      .join("");
  }

  async function set(patch) {
    Object.assign(state, patch);
    await Promise.all([renderStrip(), renderReadout()]);
    listeners.forEach((fn) => fn(state));
  }

  new ResizeObserver(() => renderReadout()).observe(bubbles);
  return { set, onChange: (fn) => listeners.add(fn) };
}

function offsetOf(manifest, videoId) {
  let offset = 0;
  for (const v of manifest.videos) {
    if (v.id === videoId) return offset;
    offset += v.segments;
  }
  return offset;
}

export { offsetOf };
