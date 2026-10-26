"""Experiment 4: can the embedding tell a simulator from a camera?

Three videos are rendered (F-16 in DCS, Assetto Corsa, the Chrome T-Rex game) and three are real
(two alpine climbs, a kart race). The cleanest pair is kart (real) vs Assetto (sim): both are
first-person racing.

1. Text probe: score "video game footage" minus "real camera footage" (averaged over a few
   phrasings) per segment and measure how well it separates sim from real (ROC AUC).
2. Learned direction: the difference between the mean sim and mean real video vectors, read out
   as words. What does the model think separates the two?

    uv run python -m experiments.sim_real qwen3vl gemma2 siglip2
"""

import sys

import numpy as np

from experiments.common import model, save, segments, space, table, world_of
from lens.readout import unit

SIM = ["video game footage", "computer graphics", "a screenshot of a simulator game"]
REAL = ["real camera footage", "a real photograph", "footage filmed with a camera in real life"]


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative."""
    wins = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(wins + 0.5 * ties)


def run(name: str) -> dict:
    keys, videos = segments(name)
    m = model(name)
    probe = m.embed_texts(SIM).mean(axis=0) - m.embed_texts(REAL).mean(axis=0)
    scores = videos @ probe
    is_sim = np.array([world_of(k) == "sim" for k in keys])
    video = np.array([k.split(":")[0] for k in keys])

    # Words along the learned sim-vs-real direction, in the centred space
    s = space(name)
    direction = unit(videos[is_sim].mean(axis=0) - videos[~is_sim].mean(axis=0))
    along = s.vocab.centered @ direction
    order = np.argsort(along)
    return {
        "model": name,
        "auc_all": auc(scores[is_sim], scores[~is_sim]),
        "auc_racing": auc(scores[video == "assetto"], scores[video == "kart"]),
        "sim_words": [s.phrases[i] for i in order[::-1][:12]],
        "real_words": [s.phrases[i] for i in order[:12]],
    }


if __name__ == "__main__":
    results = [run(name) for name in sys.argv[1:]]
    save("sim_real", results)
    print(table(results, ["model", "auc_all", "auc_racing"]))
    for r in results:
        print(f"\n{r['model']}\n  sim side:  {', '.join(r['sim_words'])}\n  real side: {', '.join(r['real_words'])}")
