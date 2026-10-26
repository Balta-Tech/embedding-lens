"""Experiment 1: which score correction gives the most faithful words?

For every model, vocabulary size and read-out method, take each segment's top 5 words and measure:
round-trip retrieval (do the words find their segment again?), hubness (do the same words show up
everywhere?) and a second opinion from another model.

    uv run python -m experiments.methods siglip2 gemma2 qwen3vl
"""

import sys

from experiments.common import hubness, round_trip, save, second_opinion, segments, space, table
from lens.readout import METHODS
from lens.vocab import TIERS

TOP = 5
# Who judges whom: each model is checked by the strongest other model available
JUDGE = {"qwen3vl": "gemma2", "gemma2": "qwen3vl", "siglip2": "qwen3vl"}


def run(models: list[str]) -> list[dict]:
    rows = []
    for name in models:
        _, videos = segments(name)
        for tier in TIERS:
            s = space(name, tier)
            for method in METHODS:
                if method == "neutral":
                    continue  # same ranking as raw by construction (see lens/readout.py)
                words = [[p for p, _ in s.top(v, TOP, method)] for v in videos]
                row = {"model": name, "vocab": tier, "method": method}
                row |= round_trip(name, words) | hubness(words)
                judge = JUDGE[name]
                row["judge"] = second_opinion(judge, words) if judge in models else float("nan")
                rows.append(row)
                print(table([row], list(row)), flush=True)
    return rows


if __name__ == "__main__":
    models = sys.argv[1:]
    rows = run(models)
    save("methods", rows)
    print()
    print(table(rows, list(rows[0])))
