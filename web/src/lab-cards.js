// One card per experiment. Takeaways are computed from the results, so they stay true on re-runs.
import { bars, card, registerCard } from "./lab.js";

const METHOD_LABELS = {
  raw: "raw cosine",
  null: "null-video",
  prior: "phrase prior",
  zscore: "z-score",
  csls: "CSLS",
  centered: "centred",
  centered_csls: "centred + CSLS",
};
const TINTS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)", "var(--faint)"];
const pct = (v) => `${Math.round(v * 100)}%`;

registerCard(async ({ loadResult, labels }) => {
  const rows = await loadResult("methods");
  if (!rows) return null;
  const models = [...new Set(rows.map((r) => r.model))];
  const tier = "medium";
  const at = (m, method) => rows.find((r) => r.model === m && r.vocab === tier && r.method === method);
  const methods = Object.keys(METHOD_LABELS).filter((k) => rows.some((r) => r.method === k));

  // Faithful = another model agrees the words fit; specific = the words find their moment again.
  const body = document.createElement("div");
  const table = document.createElement("table");
  table.innerHTML = `<thead><tr><th>Read-out</th>${models.map((m) => `<th>${labels[m] || m}<br>agree · find</th>`).join("")}</tr></thead>`;
  const tbody = document.createElement("tbody");
  const bestJudge = Object.fromEntries(models.map((m) => [m, Math.max(...methods.map((k) => at(m, k)?.judge ?? -1))]));
  for (const k of methods) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${METHOD_LABELS[k]}</td>` + models.map((m) => {
      const r = at(m, k);
      if (!r) return "<td>–</td>";
      const best = r.judge === bestJudge[m] ? "best" : "";
      return `<td class="${best}">${Number.isNaN(r.judge) ? "–" : r.judge.toFixed(1)} · ${pct(r["r@5"])}</td>`;
    }).join("");
    tbody.append(tr);
  }
  table.append(tbody);
  body.append(table);

  const main = models[0];
  const winner = methods.reduce((a, b) => ((at(main, a)?.judge ?? -1) >= (at(main, b)?.judge ?? -1) ? a : b));
  return card({
    title: "Which correction gives honest words?",
    takeaway: `Each moment's top 5 words, checked two ways: <b>agree</b> is how highly a second model ranks them (percentile), <b>find</b> is how often the words alone find their moment again among 291 (top 5). For ${labels[main] || main}, ${METHOD_LABELS[winner]} wins on agreement.`,
    wide: true,
    body,
  });
});

registerCard(async ({ loadResult, labels }) => {
  const results = await loadResult("time_arrow");
  if (!results) return null;
  const body = document.createElement("div");
  body.append(bars(results.map((r, i) => ({ label: labels[r.model] || r.model, value: 1 - r.reversed_cos, tint: TINTS[i] })), {
    max: Math.max(0.05, ...results.map((r) => 1 - r.reversed_cos)),
    format: (v) => v.toFixed(3),
  }));
  const flips = results.map((r) => {
    const off = r.probe.filter((p) => p.phase === "take-off");
    const flipped = off.filter((p) => Math.sign(p.forward) !== Math.sign(p.backward)).length;
    return `${labels[r.model] || r.model}: ${flipped}/${off.length} take-offs read as landings when reversed`;
  });
  const note = document.createElement("p");
  note.className = "hint";
  note.innerHTML = flips.join("<br>");
  body.append(note);
  return card({
    title: "Does it know which way time runs?",
    takeaway: "How far a moment's vector moves when its frames play backwards (0 = time-blind). An averaged image model cannot tell; a video model might.",
    body,
  });
});

registerCard(async ({ loadResult, labels }) => {
  const results = await loadResult("sim_real");
  if (!results) return null;
  const body = document.createElement("div");
  body.append(bars(results.flatMap((r, i) => [
    { label: `${labels[r.model] || r.model}: all`, value: r.auc_all, tint: TINTS[i] },
    { label: "kart vs Assetto", value: r.auc_racing, tint: TINTS[i] },
  ]), { max: 1, format: (v) => v.toFixed(2) }));
  const best = results.reduce((a, b) => (a.auc_racing >= b.auc_racing ? a : b));
  const words = document.createElement("div");
  words.className = "words-pair";
  words.innerHTML = `<div><b>Rendered side</b>${best.sim_words.slice(0, 8).join(", ")}</div><div><b>Camera side</b>${best.real_words.slice(0, 8).join(", ")}</div>`;
  body.append(words);
  return card({
    title: "Simulator or camera?",
    takeaway: `How well “video game footage” vs “real camera footage” separates rendered from filmed moments (ROC AUC, 0.5 = chance). Hardest pair: real kart racing vs Assetto Corsa. Below: what ${labels[best.model] || best.model} thinks separates them.`,
    body,
  });
});

registerCard(async ({ loadResult, labels }) => {
  const results = await loadResult("residual");
  if (!results) return null;
  const body = document.createElement("div");
  body.append(bars(results.flatMap((r, i) => [
    { label: `${labels[r.model] || r.model}: content`, value: r.action_share_content, tint: TINTS[i] },
    { label: "only-in-motion", value: r.action_share_residual, tint: TINTS[i] },
  ]), { max: 1, format: pct }));
  const r = results[0];
  const ex = r.examples.find((e) => e.segment.startsWith("zugspitze")) || r.examples[0];
  const words = document.createElement("div");
  words.className = "words-pair";
  words.innerHTML = `<div><b>${ex.segment}: what's in it</b>${ex.content.slice(0, 6).join(", ")}</div><div><b>what only motion adds</b>${ex.motion.slice(0, 6).join(", ")}</div>`;
  body.append(words);
  return card({
    title: "What frames alone don't show",
    takeaway: "The video vector minus the average of its own frames, read out as words. Share of action words (verbs, Kinetics actions) in the normal read-out vs in this motion-only residual.",
    body,
  });
});

registerCard(async ({ loadResult, labels }) => {
  const rows = await loadResult("templates");
  if (!rows) return null;
  const body = document.createElement("div");
  const sel = rows.filter((r) => r.method === "centered");
  body.append(bars(sel.map((r, i) => ({
    label: `${(labels[r.model] || r.model).split(" ")[0]}: ${r.template.replace("{}", "…")}`,
    value: r["r@5"],
    tint: TINTS[[...new Set(sel.map((x) => x.model))].indexOf(r.model)],
  })), { max: 1, format: pct }));
  return card({
    title: "Does the prompt around a word matter?",
    takeaway: "Each phrase wrapped in a template before embedding (“a video of …”), or averaged over all of them. Round-trip retrieval of the top 5 words, small vocabulary.",
    body,
  });
});

registerCard(async ({ loadResult, labels }) => {
  const rows = await loadResult("sparse");
  if (!rows) return null;
  const body = document.createElement("table");
  body.innerHTML = `<thead><tr><th>Model</th><th>Read-out</th><th>find (top 5)</th><th>redundancy</th><th>fit</th></tr></thead><tbody>${rows.map((r) =>
    `<tr><td>${labels[r.model] || r.model}</td><td>${r["read-out"]}</td><td>${pct(r["r@5"])}</td><td>${r.redundancy.toFixed(2)}</td><td>${r.fit.toFixed(2)}</td></tr>`).join("")}</tbody>`;
  return card({
    title: "Top words vs a sparse sum",
    takeaway: "Top-k returns near-synonyms; a sparse decomposition picks words that each explain something new. Redundancy is the mean similarity between the chosen words; fit is how much of the vector they reproduce.",
    body,
  });
});
