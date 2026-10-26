"""Experiment 6: reading out what *changed* between two moments.

The difference between consecutive segment vectors is itself a direction in the shared space.
Reading it out with the vocabulary gives words for what appeared ("glacier", "crampons") and,
on the negative side, what went away. Large changes mark scene cuts.

    uv run python -m experiments.delta qwen3vl zugspitze
"""

import sys
from itertools import pairwise

from experiments.common import save, segments, space
from lens.readout import change

TOP = 6


def run(name: str, video: str) -> list[dict]:
    keys, vectors = segments(name)
    idx = [i for i, k in enumerate(keys) if k.startswith(video + ":")]
    s = space(name, "medium")
    rows = []
    for a, b in pairwise(idx):
        amount, appeared, faded = change(s, vectors[a], vectors[b], TOP)
        rows.append({"from": keys[a], "to": keys[b], "change": amount,
                     "appeared": appeared, "faded": faded})
    return rows


if __name__ == "__main__":
    name, video = sys.argv[1], sys.argv[2]
    rows = run(name, video)
    save(f"delta_{name}_{video}", rows)
    for r in sorted(rows, key=lambda r: -r["change"])[:12]:
        print(f"{r['from']:>14} -> {r['to']:<14} change {r['change']:.2f}")
        print(f"    + {', '.join(r['appeared'])}\n    - {', '.join(r['faded'])}")
