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
}, 4);

registerCard(async ({ loadResult, labels }) => {
  const results = await loadResult("time_arrow");
  if (!results) return null;
  const body = document.createElement("div");
  body.append(bars(results.map((r, i) => ({ label: labels[r.model] || r.model, value: 1 - r.reversed_cos, tint: TINTS[i] })), {
    max: Math.max(0.05, ...results.map((r) => 1 - r.reversed_cos)),
    format: (v) => v.toFixed(3),
  }));
  // Played backwards, a take-off should read more like a landing and vice versa.
  // Shift toward the opposite event, averaged over both phases: > 0 means time-aware.
  const shift = (r) => {
    const s = r.probe.map((p) => (p.phase === "take-off" ? p.forward - p.backward : p.backward - p.forward));
    return s.reduce((a, b) => a + b, 0) / s.length;
  };
  const note = document.createElement("p");
  note.className = "hint";
  note.innerHTML = "Played backwards, does a take-off read more like a landing? Average shift toward the opposite event:<br>" +
    results.map((r) => `${labels[r.model] || r.model}: ${shift(r) >= 0 ? "+" : ""}${shift(r).toFixed(4)}`).join("<br>");
  body.append(note);
  return card({
    title: "Does it know which way time runs?",
    takeaway: "How far a moment's vector moves when its frames play backwards (0 = time-blind). An averaged image model cannot tell; a video model might.",
    body,
  });
}, 7);

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
}, 9);

registerCard(async ({ loadResult, labels }) => {
  const results = await loadResult("residual");
  if (!results) return null;
  const body = document.createElement("div");
  body.append(bars(results.map((r, i) => ({
    label: labels[r.model] || r.model,
    value: 1 - r.video_vs_frames_cos,
    tint: TINTS[i],
  })), { max: Math.max(0.05, ...results.map((r) => 1 - r.video_vs_frames_cos)), format: (v) => v.toFixed(2) }));
  // The most telling example: a sequence-only residual that names the kind of footage
  const pick = results
    .flatMap((r) => r.examples.map((e) => ({ ...e, model: r.model })))
    .find((e) => e.motion.slice(0, 4).some((w) => /video game|screenshot|simulator/.test(w)))
    || results[0].examples[0];
  const words = document.createElement("div");
  words.className = "words-pair";
  words.innerHTML = `<div><b>${pick.segment}: what's in it</b>${pick.content.slice(0, 5).join(", ")}</div><div><b>what only the sequence adds</b>${pick.motion.slice(0, 5).join(", ")}</div>`;
  body.append(words);
  return card({
    title: "What frames alone don't show",
    takeaway: `How far a model's video vector sits from the average of its own frames (0 = a bag of frames). The leftover, read out as words, is mostly noise and not actions (action words: ${results.map((r) => `${pct(r.action_share_residual)} vs ${pct(r.action_share_content)} normally`).join("; ")}); occasionally it names the kind of footage.`,
    body,
  });
}, 8);

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
}, 6);

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
}, 5);

registerCard(async ({ loadResult, labels }) => {
  const rows = await loadResult("eyeball_score");
  if (!rows) return null;
  const models = [...new Set(rows.map((r) => r.model))];
  const methods = ["raw", "centered", "centered_csls", "prior", "sparse"];
  const tiers = ["small", "", "large"];
  const body = document.createElement("table");
  const get = (m, method, tier) => rows.find((r) => r.model === m && r.method === (tier ? `${method}/${tier}` : method));
  body.innerHTML = `<thead><tr><th>Read-out</th>${models.map((m) => `<th>${labels[m] || m}<br>4.6k · 23k · 119k</th>`).join("")}</tr></thead><tbody>${methods.map((k) => `
    <tr><td>${METHOD_LABELS[k]}</td>${models.map((m) => {
      const best = Math.max(...rows.filter((r) => r.model === m).map((r) => r["precision@5"]));
      return `<td>${tiers.map((t) => {
        const r = get(m, k, t);
        if (!r) return "–";
        const v = Math.round(r["precision@5"] * 100);
        return r["precision@5"] === best ? `<b style="color:var(--good)">${v}</b>` : v;
      }).join(" · ")}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
  return card({
    title: "Checked by eye",
    takeaway: "Share of top-5 words that are actually visible in 12 sample moments (judged against the frames by an AI annotator; the list is in experiments/judgements.json). By vocabulary size: small · medium · large. Bigger vocabularies let rare near-matches in; centring degrades least.",
    wide: true,
    body,
  });
}, 1);

registerCard(async ({ loadResult, labels }) => {
  const rows = await loadResult("geometry");
  if (!rows) return null;
  const raw = rows.filter((r) => r.method === "raw");
  const body = document.createElement("div");
  body.append(bars(raw.map((r, i) => ({ label: labels[r.model] || r.model, value: r.cross_mean, tint: TINTS[i] })), {
    max: 1, format: (v) => v.toFixed(2),
  }));
  const note = document.createElement("p");
  note.className = "hint";
  note.innerHTML = Object.entries(rows.reduce((acc, r) => {
    (acc[r.model] ||= []).push(`${METHOD_LABELS[r.method]} ${pct(r.hub_top)}`);
    return acc;
  }, {})).map(([m, parts]) => `${labels[m] || m}: top-10 words take ${parts.join(", ")} of all slots`).join("<br>");
  body.append(note);
  return card({
    title: "Why the best correction depends on the model",
    takeaway: "Mean cosine between any video and any phrase. When every video is already close to every phrase, a few generic words win everywhere (hub share below) and centring helps; when the two sit far apart, raw closeness is already well calibrated.",
    body,
  });
}, 3);

registerCard(async ({ loadResult, labels }) => {
  const data = await loadResult("speed");
  if (!data) return null;
  const body = document.createElement("table");
  body.innerHTML = `<thead><tr><th>Model</th><th>phrases / s</th><th>4.6k</th><th>23k</th><th>119k</th><th>moments / s</th></tr></thead><tbody>${data.rows.map((r) =>
    `<tr><td>${labels[r.model] || r.model}</td><td>${Math.round(r.phrases_per_s)}</td><td>${r.small_min.toFixed(1)} min</td><td>${r.medium_min.toFixed(1)} min</td><td>${r.large_min.toFixed(0)} min</td><td>${r.segments_per_s.toFixed(1)}</td></tr>`).join("")}</tbody>`;
  return card({
    title: "How slow is it?",
    takeaway: "Measured on an Apple M3 Pro laptop (18 GB) while indexing. Vocabularies are embedded once and cached; reading out a moment afterwards is a single matrix product and takes milliseconds.",
    body,
  });
}, 10);

registerCard(async ({ loadResult }) => {
  const data = await loadResult("agreement");
  if (!data) return null;
  const body = document.createElement("div");
  body.className = "bars";
  body.innerHTML = Object.entries(data.correlations).map(([name, rho]) => {
    const width = Math.abs(rho) * 50;
    const left = rho >= 0 ? 50 : 50 - width;
    const tint = rho >= 0 ? "var(--good)" : "var(--c5)";
    return `<div class="row"><span>${name}</span>
      <div class="track" style="position:relative"><span style="position:absolute;left:${left}%;width:${width}%;background:${tint}"></span><i style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--faint)"></i></div>
      <span class="val">${rho >= 0 ? "+" : ""}${rho.toFixed(2)}</span></div>`;
  }).join("");
  return card({
    title: "The obvious metric points the wrong way",
    takeaway: `Rank correlation with precision judged by eye, over ${data.configurations} combinations of model, read-out and vocabulary. Words that find their moment again are often wrong; words another model agrees with are usually right.`,
    body,
  });
}, 2);
