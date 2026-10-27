// You vs the model: describe a moment, then see where your words rank in the model's eyes.
import { backend, formatTime, liveScore, loadGameRanks, loadGameVocab, loadReadouts, spriteUrl, timeOf } from "./data.js";
import { busy, waiting } from "./wait.js";

const ROUNDS = 5;
const MAX_WORDS = 5;

function shuffle(items) {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Typed words are matched to the vocabulary loosely: case, plurals, hyphens vs spaces
function variants(word) {
  const w = word.trim().toLowerCase().replace(/\s+/g, " ");
  const forms = [w, w.replace(/-/g, " "), w.replace(/ /g, "-"), w.replace(/[- ]/g, "")];
  for (const f of [...forms]) {
    if (f.endsWith("ies")) forms.push(`${f.slice(0, -3)}y`);
    if (f.endsWith("es")) forms.push(f.slice(0, -2));
    if (f.endsWith("s")) forms.push(f.slice(0, -1));
  }
  return forms;
}

function distance(a, b, limit) {
  // Levenshtein, giving up once every path exceeds the limit
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const row = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      best = Math.min(best, row[j]);
    }
    if (best > limit) return limit + 1;
    prev = row;
  }
  return prev[b.length];
}

// Closest vocabulary entries to a word that isn't in it: completions first, then near spellings
function closest(word, vocab, n = 3) {
  const w = word.trim().toLowerCase();
  const limit = w.length < 5 ? 1 : 2;
  const scored = [];
  for (const v of vocab) {
    if (v.startsWith(w)) scored.push([v.length - w.length - 100, v]);
    else if (Math.abs(v.length - w.length) <= limit) {
      const d = distance(w, v, limit);
      if (d <= limit) scored.push([d, v]);
    }
  }
  return scored.sort((a, b) => a[0] - b[0]).slice(0, n).map(([, v]) => v);
}

export function createGame(manifest) {
  const root = document.getElementById("game-root");
  // The game is always played against the default (most faithful) model
  const model = manifest.defaultModel;
  const modelLabel = manifest.models.find((m) => m.id === model)?.label ?? model;
  let game;

  async function start() {
    const [data, readouts] = await Promise.all([loadGameVocab(model), loadReadouts(model, "medium")]);
    const index = new Map(data.vocab.map((w, i) => [w, i]));
    // Skip the first and last moment of each video (intros, end screens) and frozen frames
    const pool = manifest.videos.flatMap((v) =>
      Array.from({ length: v.segments - 2 }, (_, i) => `${v.id}:${i + 1}`)
        .filter((k) => readouts.segments[k]?.change?.appeared?.length !== 0));
    game = { rounds: shuffle(pool).slice(0, ROUNDS), round: 0, total: 0, words: [], data, index, readouts };
    renderRound();
  }

  function renderRound() {
    if (game.round >= ROUNDS) return renderFinal();
    const key = game.rounds[game.round];
    game.words = [];
    root.innerHTML = `
      <div>
        <div class="game-head"><span>Round ${game.round + 1} / ${ROUNDS}</span><span class="game-score">${Math.round(game.total)} pts</span></div>
        <div class="game-frame"><div class="sprite" style="background-image:url(${spriteUrl(key)})"></div></div>
        <p class="hint">${manifest.videos.find((v) => v.id === key.split(":")[0]).title} · ${formatTime(timeOf(manifest, key))}</p>
      </div>
      <div>
        <label class="control-label" for="game-input" style="width:auto">Your words</label>
        <div class="word-entry" id="word-entry">
          <input id="game-input" placeholder="type a word, press Enter" autocomplete="off" />
        </div>
        <p class="suggest" id="game-suggest" hidden></p>
        <div class="game-actions">
          <button class="btn primary" id="reveal" type="button" disabled>Reveal</button>
          <button class="btn ghost" id="skip" type="button">Skip</button>
        </div>
        <div id="game-result"></div>
      </div>`;
    const input = root.querySelector("#game-input");
    let timer;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => suggest(input.value), 150);
    });
    input.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === ",") && input.value.trim()) {
        e.preventDefault();
        addWord(input.value);
        input.value = "";
        suggest("");
      } else if (e.key === "Backspace" && !input.value && game.words.length) {
        game.words.pop();
        renderChips();
      }
    });
    root.querySelector("#reveal").addEventListener("click", () => {
      if (input.value.trim()) addWord(input.value);
      reveal(key);
    });
    root.querySelector("#skip").addEventListener("click", () => { game.round++; renderRound(); });
    input.focus({ preventScroll: true });
  }

  // Offline, only words in the precomputed list can be ranked: once the player has typed a word
  // that isn't in it, offer the nearest ones. Live, any word works, so no suggestions.
  function suggest(value) {
    const el = root.querySelector("#game-suggest");
    const known = (w) => variants(w).some((f) => game.index.has(f));
    if (backend.live || value.trim().length < 3 || known(value)) {
      el.hidden = true;
      return;
    }
    const options = closest(value, game.data.vocab);
    el.hidden = false;
    el.innerHTML = options.length
      ? `Not in the offline word list. Closest: ${options.map((o) => `<button type="button" class="chip">${o}</button>`).join("")}`
      : "Not in the offline word list, so it can't be ranked without the live backend.";
    el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      const input = root.querySelector("#game-input");
      addWord(b.textContent);
      input.value = "";
      el.hidden = true;
      input.focus();
    }));
  }

  function addWord(raw) {
    const w = raw.trim().toLowerCase().replace(/,$/, "");
    if (!w || game.words.includes(w) || game.words.length >= MAX_WORDS) return;
    game.words.push(w);
    renderChips();
  }

  function renderChips() {
    const entry = root.querySelector("#word-entry");
    entry.querySelectorAll(".word-chip").forEach((c) => c.remove());
    const input = entry.querySelector("input");
    game.words.forEach((w, i) => {
      const chip = document.createElement("span");
      chip.className = "word-chip";
      chip.innerHTML = `${w}<button type="button" aria-label="remove ${w}">×</button>`;
      chip.querySelector("button").addEventListener("click", () => { game.words.splice(i, 1); renderChips(); });
      entry.insertBefore(chip, input);
    });
    input.disabled = game.words.length >= MAX_WORDS;
    root.querySelector("#reveal").disabled = game.words.length === 0;
  }

  async function score(key) {
    if (backend.live) {
      try {
        const r = await liveScore(key, game.words, model);
        return { size: r.vocabulary_size, words: r.words.map((w) => ({ phrase: w.phrase, rank: w.rank, percentile: w.percentile })) };
      } catch { /* fall back to the offline table */ }
    }
    const size = game.data.vocab.length;
    const ranks = await loadGameRanks(model, key);
    return {
      size,
      words: game.words.map((w) => {
        const i = variants(w).map((f) => game.index.get(f)).find((x) => x !== undefined);
        if (i === undefined) return { phrase: w, rank: null, percentile: null };
        const rank = ranks[i];
        return { phrase: w, rank, percentile: 100 * (1 - (rank - 1) / size) };
      }),
    };
  }

  async function reveal(key) {
    const button = root.querySelector("#reveal");
    root.querySelector("#game-input").disabled = true;
    root.querySelector("#game-suggest").hidden = true;
    let result;
    if (backend.live) {
      const done = busy(button, "Scoring…");
      const out = root.querySelector("#game-result");
      out.innerHTML = '<p class="hint"></p>';
      waiting(out.firstChild, "The live model is ranking your words among 119,000. This takes a few seconds.");
      result = await score(key);
      done();
    } else {
      result = await score(key);
    }
    button.disabled = true;
    const readouts = game.readouts;
    const known = result.words.filter((w) => w.percentile !== null).map((w) => w.percentile).sort((a, b) => b - a);
    // Round score: the average of your best three words, so one stray guess doesn't sink you
    const best = known.slice(0, 3);
    const points = best.length ? best.reduce((a, b) => a + b, 0) / 3 : 0;
    game.total += points;
    const top = readouts.segments[key][game.data.method === "centered" ? "centered" : "raw"].slice(0, 8).map(([p]) => p);
    root.querySelector("#game-result").innerHTML = `
      <div class="verdicts">
        ${result.words.map((w) => w.percentile === null
          ? `<div class="verdict unknown"><span>${w.phrase}</span><div class="bar"></div><span class="num">not in the word list</span></div>`
          : `<div class="verdict"><span>${w.phrase}</span><div class="bar"><span style="width:${w.percentile}%"></span></div><span class="num">top ${topPct(w.rank, result.size)}</span></div>`).join("")}
      </div>
      <p class="hint">Ranked by ${modelLabel} among ${result.size.toLocaleString()} words${backend.live ? " (live)" : ""}. Round: <b>${Math.round(points)}</b> pts.</p>
      <div class="model-says"><h4>The model's own words</h4><div class="chips">${top.map((p) => `<span class="chip" aria-pressed="${game.words.includes(p)}">${p}</span>`).join("")}</div></div>
      <div class="game-actions"><button class="btn primary" id="next" type="button">${game.round + 1 < ROUNDS ? "Next moment" : "See result"} →</button></div>`;
    root.querySelector(".game-score").textContent = `${Math.round(game.total)} pts`;
    root.querySelector("#next").addEventListener("click", () => { game.round++; renderRound(); });
  }

  function renderFinal() {
    const avg = game.total / ROUNDS;
    const verdict = avg > 90 ? "You see what the model sees."
      : avg > 75 ? "Close: you and the model mostly agree."
      : avg > 50 ? "You notice different things than the model does."
      : "You and the model are looking at different movies.";
    root.innerHTML = `
      <div class="game-final" style="grid-column:1/-1">
        <p class="eyebrow">Final score</p>
        <div class="big">${Math.round(game.total)}<span style="font-size:28px;color:var(--muted)"> / ${ROUNDS * 100}</span></div>
        <p class="section-sub">${verdict}</p>
        <button class="btn primary" id="again" type="button">Play again</button>
      </div>`;
    root.querySelector("#again").addEventListener("click", start);
  }

  start();
  return { restart: start };
}

function topPct(rank, size) {
  const p = (100 * rank) / size;
  return p < 1 ? `${p.toFixed(2)}%` : `${Math.round(p)}%`;
}
