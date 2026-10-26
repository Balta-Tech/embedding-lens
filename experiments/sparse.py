"""Experiment 5: top-k words vs a sparse decomposition.

Top-k tends to return near-synonyms ("glacier", "ice field", "icefall"). Greedy matching pursuit
picks words that each explain a different part of the vector. Compared on round-trip retrieval,
redundancy (mean cosine between the chosen words) and how much of the vector the words explain.

    uv run python -m experiments.sparse qwen3vl gemma2 siglip2
"""

import sys

import numpy as np

from experiments.common import round_trip, save, segments, space, table
from lens.readout import sparse_decompose, unit

TERMS = 5


def redundancy(s, words: list[str]) -> float:
    idx = [s.phrases.index(w) for w in words]
    vecs = s.vocab.centered[idx]
    sims = vecs @ vecs.T
    n = len(idx)
    return float((sims.sum() - n) / (n * (n - 1))) if n > 1 else 0.0


def fit_of(s, v, words: list[str]) -> float:
    """Cosine between the centred segment vector and its best non-negative fit by these words."""
    target = unit(v - s.video_mean)
    basis = s.vocab.centered[[s.phrases.index(w) for w in words]].T
    w, *_ = np.linalg.lstsq(basis, target, rcond=None)
    return float(unit(basis @ np.clip(w, 0, None)) @ target)


def run(name: str) -> list[dict]:
    _, videos = segments(name)
    s = space(name, "medium")
    rows = []
    for label in ("top-k", "sparse"):
        lists = []
        for v in videos:
            if label == "top-k":
                lists.append([p for p, _ in s.top(v, TERMS, "centered")])
            else:
                lists.append([p for p, _ in sparse_decompose(v, s, TERMS)[0]])
        row = {"model": name, "read-out": label}
        row |= round_trip(name, lists)
        row["redundancy"] = float(np.mean([redundancy(s, w) for w in lists]))
        row["fit"] = float(np.mean([fit_of(s, v, w) for v, w in zip(videos, lists)]))
        rows.append(row)
    return rows


if __name__ == "__main__":
    rows = [row for name in sys.argv[1:] for row in run(name)]
    save("sparse", rows)
    print(table(rows, list(rows[0])))
