"""Experiment 2: does the prompt wrapped around each phrase matter?

The small vocabulary is re-embedded with each template, and with the mean over all templates
(prompt ensembling, as in CLIP), then read out with the centred method.

    uv run python -m experiments.templates qwen3vl gemma2 siglip2
"""

import sys

import numpy as np

from experiments.common import hubness, model, round_trip, save, segments, table
from lens.readout import Space, unit
from lens.vocab import load as load_vocab

TEMPLATES = ["{}", "a video of {}", "a photo of {}", "a first-person video showing {}"]
TOP = 5


def run(name: str) -> list[dict]:
    m = model(name)
    phrases = load_vocab("small")
    _, videos = segments(name)
    embedded = {t: m.embed_texts(phrases, template=t) for t in TEMPLATES}
    variants = dict(embedded)
    variants["ensemble"] = unit(np.mean(list(embedded.values()), axis=0))
    variants["model default"] = m.embed_texts(phrases)
    rows = []
    for label, text in variants.items():
        s = Space(phrases, text, videos)
        for method in ("raw", "centered"):
            words = [[p for p, _ in s.top(v, TOP, method)] for v in videos]
            row = {"model": name, "template": label, "method": method}
            rows.append(row | round_trip(name, words) | hubness(words))
            print(table([rows[-1]], list(rows[-1])), flush=True)
    return rows


if __name__ == "__main__":
    rows = [row for name in sys.argv[1:] for row in run(name)]
    save("templates", rows)
    print()
    print(table(rows, list(rows[0])))
