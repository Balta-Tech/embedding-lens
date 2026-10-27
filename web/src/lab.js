// Lab: the experiment results behind the defaults, rendered from data/results/*.json.
import { loadResult } from "./data.js";

const CARDS = [];

// Lower rank renders first
export function registerCard(fn, rank = 50) {
  CARDS.push({ fn, rank });
}

export async function createLab(manifest) {
  const root = document.getElementById("lab-root");
  const labels = Object.fromEntries(manifest.models.map((m) => [m.id, m.label]));
  CARDS.sort((a, b) => a.rank - b.rank);
  for (const { fn } of CARDS) {
    const el = await fn({ loadResult, labels });
    if (el) root.append(el);
  }
  if (!root.children.length) {
    root.innerHTML = `<p class="hint">Experiment results will appear here.</p>`;
  }
}

export function card({ title, takeaway, wide = false, body }) {
  const el = document.createElement("article");
  el.className = `card${wide ? " wide" : ""}`;
  el.innerHTML = `<h3>${title}</h3><p class="takeaway">${takeaway}</p>`;
  el.append(body);
  return el;
}

export function bars(rows, { max = 1, format = (v) => v.toFixed(2), color = "var(--c1)" } = {}) {
  const el = document.createElement("div");
  el.className = "bars";
  el.innerHTML = rows
    .map(({ label, value, tint }) => `
      <div class="row"><span>${label}</span>
        <div class="track"><span style="width:${Math.max(0, Math.min(100, (100 * value) / max))}%;background:${tint || color}"></span></div>
        <span class="val">${format(value)}</span></div>`)
    .join("");
  return el;
}
