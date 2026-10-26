// You vs the model: describe a moment, then see where your words rank in the model's eyes.
import { backend, formatTime, liveScore, loadGame, loadReadouts, spriteUrl, timeOf } from "./data.js";

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

export function createGame(manifest, state) {
  const root = document.getElementById("game-root");
  let game;

  async function start() {
    const data = await loadGame(state.model);
    // Skip the first and last moment of each video: intros and end screens
    const pool = data.keys.filter((k) => {
      const [v, i] = k.split(":");
      const video = manifest.videos.find((x) => x.id === v);
      return Number(i) > 0 && Number(i) < video.segments - 1;
    });
    game = { rounds: shuffle(pool).slice(0, ROUNDS), round: 0, total: 0, words: [], data };
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
          <input id="game-input" list="game-vocab" placeholder="type a word, press Enter" autocomplete="off" />
        </div>
        <datalist id="game-vocab"></datalist>
        <div class="game-actions">
          <button class="btn primary" id="reveal" type="button" disabled>Reveal</button>
          <button class="btn ghost" id="skip" type="button">Skip</button>
        </div>
        <div id="game-result"></div>
      </div>`;
    const input = root.querySelector("#game-input");
    root.querySelector("#game-vocab").innerHTML = game.data.vocab.map((w) => `<option value="${w}">`).join("");
    input.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === ",") && input.value.trim()) {
        e.preventDefault();
        addWord(input.value);
        input.value = "";
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
        const r = await liveScore(key, game.words, state.model);
        return { size: r.vocabulary_size, words: r.words.map((w) => ({ phrase: w.phrase, rank: w.rank, percentile: w.percentile })) };
      } catch { /* fall back to the offline table */ }
    }
    const { vocab, keys, ranks } = game.data;
    const row = keys.indexOf(key) * vocab.length;
    return {
      size: vocab.length,
      words: game.words.map((w) => {
        const i = vocab.indexOf(w);
        if (i < 0) return { phrase: w, rank: null, percentile: null };
        const rank = ranks[row + i];
        return { phrase: w, rank, percentile: 100 * (1 - (rank - 1) / vocab.length) };
      }),
    };
  }

  async function reveal(key) {
    root.querySelector("#reveal").disabled = true;
    root.querySelector("#game-input").disabled = true;
    const [result, readouts] = await Promise.all([score(key), loadReadouts(state.model, "small")]);
    const known = result.words.filter((w) => w.percentile !== null).map((w) => w.percentile).sort((a, b) => b - a);
    // Round score: the average of your best three words, so one stray guess doesn't sink you
    const best = known.slice(0, 3);
    const points = best.length ? best.reduce((a, b) => a + b, 0) / 3 : 0;
    game.total += points;
    const top = readouts.segments[key][game.data.method === "centered" ? "centered" : "raw"].slice(0, 8).map(([p]) => p);
    root.querySelector("#game-result").innerHTML = `
      <div class="verdicts">
        ${result.words.map((w) => w.percentile === null
          ? `<div class="verdict unknown"><span>${w.phrase}</span><div class="bar"></div><span class="num">not in offline list</span></div>`
          : `<div class="verdict"><span>${w.phrase}</span><div class="bar"><span style="width:${w.percentile}%"></span></div><span class="num">top ${topPct(w.rank, result.size)}</span></div>`).join("")}
      </div>
      <p class="hint">Ranked among ${result.size.toLocaleString()} words${backend.live ? " (live)" : ""}. Round: <b>${Math.round(points)}</b> pts.</p>
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
