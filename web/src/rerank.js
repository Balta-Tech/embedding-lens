// Rerank: the embedding proposes, a reranker reads video and phrase together and re-sorts.
// Drawn as a slope chart from the embedding's order to the reranker's, for the moment
// selected in Explore.
import { backend, formatTime, liveRerank, loadReadouts, loadRerank, spriteUrl, timeOf } from "./data.js";
import { busy, waiting } from "./wait.js";

const ROWS = 8;
const ROW = 30;

export function createRerank(manifest, state) {
  const root = document.getElementById("rerank-root");
  const chart = document.getElementById("rerank-chart");
  const meta = document.getElementById("rerank-meta");
  const liveBtn = document.getElementById("rerank-live");
  let current = null;

  async function render() {
    const key = `${state.video}:${state.index}`;
    const model = manifest.models.find((m) => m.id === state.model);
    const lens = model.bestLens;
    document.getElementById("rerank-sprite").style.backgroundImage = `url(${spriteUrl(key)})`;
    meta.innerHTML = `<b>${manifest.videos.find((v) => v.id === state.video).title.split(",")[0]} · ${formatTime(timeOf(manifest, key))}</b>
      ${model.label}, ${state.tier === "large" ? "119k" : state.tier === "medium" ? "23k" : "4.6k"} words`;
    const [readouts, reranked] = await Promise.all([loadReadouts(state.model, state.tier), loadRerank(state.model, state.tier)]);
    const before = readouts.segments[key][lens].map(([p]) => p);
    const after = reranked?.segments[key];
    current = { key, before };
    if (!after) {
      chart.innerHTML = "";
      root.dataset.empty = "";
      document.getElementById("rerank-empty").textContent = backend.live
        ? "Not precomputed for this model and vocabulary. Rerank it live:"
        : "Not precomputed for this model and vocabulary. Switch to Qwen3-VL with 23k words in Explore, or start the live backend.";
      updateLive();
      return;
    }
    delete root.dataset.empty;
    draw(before, after, reranked.k);
    updateLive();
  }

  function draw(before, after, k) {
    // after: [phrase, reranker logit, rank in the embedding read-out]
    const width = chart.getBoundingClientRect().width || 560;
    const left = 0;
    const right = width;
    const gutter = Math.min(150, width * 0.32);
    const height = ROWS * ROW + 34;
    const rightRank = new Map(after.slice(0, ROWS).map(([p], i) => [p, i]));
    const y = (i) => 22 + i * ROW;
    const deep = height - 4;
    const lines = [];
    const labels = [];
    before.slice(0, ROWS).forEach((p, i) => {
      const j = rightRank.get(p);
      const kept = j !== undefined;
      lines.push(`<path class="${kept ? (j < i ? "up" : j > i ? "down" : "same") : "out"}" d="M${left + gutter},${y(i) - 4} C${width / 2},${y(i) - 4} ${width / 2},${kept ? y(j) - 4 : deep} ${right - gutter},${kept ? y(j) - 4 : deep}"/>`);
      labels.push(`<text class="w${kept ? "" : " faded"}" x="${left + gutter - 10}" y="${y(i)}" text-anchor="end">${p}</text>`);
    });
    after.slice(0, ROWS).forEach(([p, , from], j) => {
      const fromTop = from <= ROWS;
      if (!fromTop) {
        lines.push(`<path class="new" d="M${left + gutter},${deep} C${width / 2},${deep} ${width / 2},${y(j) - 4} ${right - gutter},${y(j) - 4}"/>`);
      }
      labels.push(`<text class="w" x="${right - gutter + 10}" y="${y(j)}">${p}${fromTop ? "" : ` <tspan class="from">from #${from}</tspan>`}</text>`);
    });
    chart.innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="Embedding order against reranked order">
        <text class="head" x="${left + gutter - 10}" y="8" text-anchor="end">Embedding</text>
        <text class="head" x="${right - gutter + 10}" y="8">Reranked (top ${k})</text>
        <g class="lines">${lines.join("")}</g>
        <text class="deeper" x="${left + gutter - 10}" y="${deep + 3}" text-anchor="end">#9–${k}</text>
        ${labels.join("")}
      </svg>`;
  }

  function updateLive() {
    liveBtn.hidden = !backend.live;
  }

  liveBtn.addEventListener("click", async () => {
    if (!current) return;
    const done = busy(liveBtn, "Reranking…");
    const note = document.getElementById("rerank-empty");
    root.dataset.empty = "";
    const stop = waiting(note, "The reranker reads the video once, then each of the 50 phrases with it. This takes a few seconds.");
    try {
      const result = await liveRerank(current.key, state.model, state.tier);
      delete root.dataset.empty;
      draw(current.before, result.ranked, result.k);
    } catch {
      note.textContent = "The backend could not rerank this moment (it needs the frames and the reranker weights).";
    } finally {
      done();
      stop();
    }
  });

  backend.listeners.add(updateLive);
  new ResizeObserver(() => current && render()).observe(chart);
  return { render };
}
