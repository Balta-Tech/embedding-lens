// A short guided tour: each step sets the page into a state and explains what to look at.
import { loadReadouts } from "./data.js";

const words = async (state, key) => (await loadReadouts(state.model, state.tier)).segments[key];
const top = async (state, key, lens, n = 3) => {
  const data = await words(state, key);
  const list = lens === "sparse" ? data.sparse.terms : data[lens];
  return list.slice(0, n).map(([p]) => `“${p}”`).join(", ");
};
const QWEN = { model: "qwen3vl", tier: "medium" };

export const STEPS = [
  {
    target: "#explore",
    title: "A moment becomes a vector",
    async body(state) {
      return `Fifteen seconds into the Zugspitze climb: a gorge at night, lit by a head torch. These five seconds are one vector, and the words closest to it are ${await top(state, "zugspitze:3", "raw")}.`;
    },
    set: { ...QWEN, video: "zugspitze", index: 3, lens: "raw" },
  },
  {
    target: ".explorer",
    title: "Closeness, read as words",
    async body(state) {
      return `On the glacier, Qwen3-VL-Embedding says ${await top(state, "zugspitze:50", "raw", 4)}. Checked against the frames, 80% of its top-5 words are actually visible: the most faithful of the three models.`;
    },
    set: { ...QWEN, video: "zugspitze", index: 50, lens: "raw" },
  },
  {
    target: "#model-picker",
    title: "Not every model is this honest",
    async body(state) {
      return `EmbeddingGemma 2 on a browser window: ${await top(state, "trex:10", "raw", 4)}. In its space every video is close to every word, so short generic words win everywhere.`;
    },
    set: { model: "gemma2", tier: "medium", video: "trex", index: 10, lens: "raw" },
  },
  {
    target: "#lens-picker",
    title: "Centring helps it",
    async body(state) {
      return `Remove the offset all video vectors share and the generic words are gone: ${await top(state, "trex:10", "centered", 4)}. Not perfect, but about the screen now. The green dot marks the most faithful lens for each model; for the other two it is raw.`;
    },
    set: { model: "gemma2", tier: "medium", video: "trex", index: 10, lens: "centered" },
  },
  {
    target: "#tier-picker",
    title: "More words, less truth",
    async body(state) {
      const small = (await loadReadouts(state.model, "small")).segments["zugspitze:88"].raw;
      const list = small.slice(0, 3).map(([p]) => `“${p}”`).join(", ");
      return `The summit, read by SigLIP 2 against all 119,000 WordNet nouns: ${await top(state, "zugspitze:88", "raw", 4)}. Against the 4.6k-word tag list: ${list}. For every model the small list was the most precise.`;
    },
    set: { model: "siglip2", tier: "large", video: "zugspitze", index: 88, lens: "raw" },
  },
  {
    target: "#change",
    title: "What changed?",
    async body(state) {
      const c = (await words(state, "zugspitze:64")).change;
      const list = (ws, sign) => ws.slice(0, 3).map((w) => `“${sign}${w}”`).join(", ");
      return `The difference between two moments is a direction too. Here the climb leaves the glacier for steel cables, and reading the difference out gives ${list(c.appeared, "+")} and ${list(c.faded, "−")}. The orange lane in the strip marks the biggest changes.`;
    },
    set: { ...QWEN, video: "zugspitze", index: 64, lens: "raw" },
  },
  {
    target: "#strip",
    title: "Simulator or reality?",
    async body(state) {
      return `An F-16 over the sea in a flight simulator: ${await top(state, "f16:18", "raw", 4)}. Telling rendered from real is harder than it looks; the Lab shows why.`;
    },
    set: { ...QWEN, video: "f16", index: 18, lens: "raw" },
  },
  {
    target: "#search",
    title: "It works backwards too",
    body: () => "Type a scene and the same space finds the moments closest to it. The lanes show where in each video the query lights up; click one to jump there.",
    set: {},
  },
  {
    target: "#game",
    title: "Your turn",
    body: () => "Describe a moment in a few words and see where they rank among everything the model could have said.",
    set: {},
  },
  {
    target: "#lab",
    title: "How we know",
    body: () => "Every default on this page comes from a small experiment, including the judgements by eye. The Lab has the numbers.",
    set: {},
  },
];

export function createTour(explorer, state) {
  const $ = (id) => document.getElementById(id);
  const el = $("tour");
  let step = 0;
  let focused = null;

  async function show(i) {
    step = i;
    const s = STEPS[i];
    if (Object.keys(s.set).length) await explorer.set(s.set);
    focused?.classList.remove("tour-focus");
    focused = document.querySelector(s.target);
    focused?.classList.add("tour-focus");
    focused?.scrollIntoView({ behavior: "smooth", block: "center" });
    $("tour-step").textContent = `${i + 1} / ${STEPS.length}`;
    $("tour-title").textContent = s.title;
    $("tour-body").textContent = await s.body(state);
    $("tour-back").disabled = i === 0;
    $("tour-next").textContent = i === STEPS.length - 1 ? "Finish" : "Next";
    el.hidden = false;
  }

  function close() {
    el.hidden = true;
    focused?.classList.remove("tour-focus");
  }

  $("tour-next").addEventListener("click", () => (step < STEPS.length - 1 ? show(step + 1) : close()));
  $("tour-back").addEventListener("click", () => step > 0 && show(step - 1));
  $("tour-close").addEventListener("click", close);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !el.hidden) close(); });
  return { start: () => show(0) };
}
