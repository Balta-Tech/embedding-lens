// Lab: the experiment results behind the defaults, rendered from data/results/*.json.
import { loadResult } from "./data.js";

const CARDS = [];

export function registerCard(fn) {
  CARDS.push(fn);
}

export async function createLab(manifest) {
  const root = document.getElementById("lab-root");
  const labels = Object.fromEntries(manifest.models.map((m) => [m.id, m.label]));
  for (const card of CARDS) {
    const el = await card({ loadResult, labels });
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
