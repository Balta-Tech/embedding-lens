// A short guided tour: each step sets the page into a state and explains what to look at.
import { loadReadouts } from "./data.js";

const top = async (state, key, lens, n = 3) => {
  const data = (await loadReadouts(state.model)).segments[key];
  const words = lens === "sparse" ? data.sparse.terms : data[lens];
  return words.slice(0, n).map(([p]) => `“${p}”`).join(", ");
};

export const STEPS = [
  {
    target: "#explore",
    title: "A moment becomes a vector",
    async body(state) {
      return `Fifteen seconds into the Zugspitze climb: a gorge tunnel lit by a head torch. One vector for these five seconds, and the words closest to it are ${await top(state, "zugspitze:3", "raw")}.`;
    },
    set: { video: "zugspitze", index: 3, lens: "raw" },
  },
  {
    target: ".explorer",
    title: "Raw closeness says what's generic",
    async body(state) {
      return `On the glacier, raw cosine picks ${await top(state, "zugspitze:50", "raw")}. Correct, but words like these also sit close to plenty of other snowy footage.`;
    },
    set: { video: "zugspitze", index: 50, lens: "raw" },
  },
  {
    target: "#lens-picker",
    title: "Calibrated says what's specific",
    async body(state) {
      return `The calibrated lens removes the offset all video vectors share and discounts words that are close to everything. Now: ${await top(state, "zugspitze:50", "calibrated")}.`;
    },
    set: { video: "zugspitze", index: 50, lens: "calibrated" },
  },
  {
    target: "#lens-picker",
    title: "Sparse says what it's made of",
    async body(state) {
      return `The sparse lens explains the vector as a sum of a few words, each adding something new: ${await top(state, "zugspitze:50", "sparse", 4)}.`;
    },
    set: { video: "zugspitze", index: 50, lens: "sparse" },
  },
  {
    target: "#strip",
    title: "Simulator or reality?",
    async body(state) {
      return `Same model, an F-16 over the sea in a flight simulator: ${await top(state, "f16:18", "calibrated")}. The Lab section measures how well each model separates rendered from real footage.`;
    },
    set: { video: "f16", index: 18, lens: "calibrated" },
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
