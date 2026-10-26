"""Experiment 9: why does the best read-out correction depend on the model?

Measures the shape of each model's shared space:

    gap            distance between the mean video vector and the mean text vector
    video_cone     mean cosine between two different videos (1 = all videos point the same way)
    text_cone      the same for phrases
    cross_mean     mean cosine between any video and any phrase
    hub_skew       skewness of k-occurrence: how often each phrase appears in some segment's
                   top 10. High = a few phrases win everywhere (Radovanović et al. 2010)
    hub_top        share of all top-10 slots taken by the 10 most frequent phrases

    uv run python -m experiments.geometry qwen3vl gemma2 siglip2
"""

import sys

import numpy as np

from experiments.common import save, segments, space, table

K = 10


def skew(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12))


def occurrences(s, videos, method: str) -> np.ndarray:
    counts = np.zeros(len(s.phrases))
    for v in videos:
        scores = s.scores(v, method)
        counts[np.argpartition(-scores, K)[:K]] += 1
    return counts


def run(name: str) -> list[dict]:
    _, videos = segments(name)
    s = space(name, "medium")
    text = s.vocab.text
    rng = np.random.default_rng(0)
    sample = text[rng.choice(len(text), 3000, replace=False)]
    vv = videos @ videos.T
    tt = sample @ sample.T
    base = {
        "model": name,
        "gap": float(np.linalg.norm(videos.mean(0) - text.mean(0))),
        "video_cone": float((vv.sum() - len(vv)) / (len(vv) * (len(vv) - 1))),
        "text_cone": float((tt.sum() - len(tt)) / (len(tt) * (len(tt) - 1))),
        "cross_mean": float((videos @ sample.T).mean()),
    }
    rows = []
    for method in ("raw", "centered", "centered_csls"):
        n = occurrences(s, videos, method)
        top = np.sort(n)[::-1][:10].sum() / n.sum()
        rows.append(base | {"method": method, "hub_skew": skew(n), "hub_top": float(top)})
    return rows


if __name__ == "__main__":
    rows = [r for name in sys.argv[1:] for r in run(name)]
    save("geometry", rows)
    print(table(rows, list(rows[0])))
