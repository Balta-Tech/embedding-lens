"""Experiment 8: read-outs side by side for a fixed sample, to judge against the frames by eye.

Automatic metrics reward words that identify a moment, including odd ones. This prints each
method's top 5 words for a dozen moments so a person can mark which words are actually visible.
Judgements are stored separately in data/results/eyeball_judgements.json.

    uv run python -m experiments.eyeball qwen3vl gemma2 siglip2
"""

import sys

from experiments.common import save, segments, space
from lens.readout import sparse_decompose

SAMPLE = [
    "zugspitze:3", "zugspitze:13", "zugspitze:28", "zugspitze:50", "zugspitze:61", "zugspitze:88",
    "wetterstein:20", "kart:20", "f16:9", "f16:18", "assetto:13", "trex:10",
]
READOUTS = {
    "raw": lambda s, v: s.top(v, 5, "raw"),
    "null": lambda s, v: s.top(v, 5, "null"),
    "prior": lambda s, v: s.top(v, 5, "prior"),
    "centered": lambda s, v: s.top(v, 5, "centered"),
    "centered_csls": lambda s, v: s.top(v, 5, "centered_csls"),
    "centered_csls@0.25": lambda s, v: s.top(v, 5, "centered_csls", 0.25),
    "sparse": lambda s, v: sparse_decompose(v, s, 5)[0],
}


def run(name: str, tier: str = "medium") -> dict:
    keys, videos = segments(name)
    s = space(name, tier)
    out = {}
    for key in SAMPLE:
        v = videos[keys.index(key)]
        out[key] = {label: [p for p, _ in fn(s, v)] for label, fn in READOUTS.items()}
    return out


if __name__ == "__main__":
    results = {name: run(name) for name in sys.argv[1:]}
    save("eyeball", results)
    for name, panels in results.items():
        print(f"\n=== {name}")
        for key, rows in panels.items():
            print(f"\n{key}")
            for label, words in rows.items():
                print(f"  {label:>18}: {', '.join(words)}")
