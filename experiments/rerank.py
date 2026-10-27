"""Experiment 14: does a reranker fix the embedding read-out?

Two stages, as in retrieval: the embedding read-out proposes its top K phrases, then
Qwen3-VL-Reranker-2B reads each (video, phrase) pair together and re-sorts them. Scored against
judgements.json (precision@5 of words judged visible), for every first-stage model and for the
medium and large vocabularies, plus a sweep over K for Qwen3-VL-Embedding.

    uv run python -m experiments.rerank
    uv run python -m experiments.rerank --review   # words no read-out produced before, to judge
"""

import json
import sys
import time
from pathlib import Path

from experiments.common import save, segments, table
from experiments.eyeball import SAMPLE
from lens.config import BEST_METHOD, DATA, RESULTS_DIR
from lens.index import load_vocab
from lens.readout import Space
from lens.rerank import Reranker
from lens.videos import all_segments

JUDGEMENTS_PATH = Path(__file__).parent / "judgements.json"
MODELS = ("qwen3vl", "siglip2", "gemma2")
TIERS = ("medium", "large")
K = 50
K_SWEEP = (10, 20, 100)


def candidates(name: str, tier: str, k: int) -> dict[str, list[str]]:
    keys, videos = segments(name)
    phrases, text = load_vocab(name, tier)
    space = Space(phrases, text, videos)
    return {key: [p for p, _ in space.top(videos[keys.index(key)], k, BEST_METHOD[name])] for key in SAMPLE}


def run() -> dict:
    reranker = Reranker()
    by_key = {s.key: s for s in all_segments()}
    out = {"k": K, "readouts": {}, "seconds_per_moment": {}}
    for name in MODELS:
        for tier in TIERS:
            ks = (K, *K_SWEEP) if (name, tier) == ("qwen3vl", "medium") else (K,)
            pool = candidates(name, tier, max(ks))
            for k in ks:
                label = f"{name}/{tier}/{k}"
                start = time.time()
                rows = {}
                for key, words in pool.items():
                    scores = reranker.score(by_key[key], words[:k])
                    order = sorted(range(k), key=lambda i: -scores[i])
                    rows[key] = {"before": words[:5], "after": [words[i] for i in order[:5]],
                                 "moved_from": [i + 1 for i in order[:5]]}
                out["readouts"][label] = rows
                out["seconds_per_moment"][label] = (time.time() - start) / len(pool)
                print(f"{label}: {out['seconds_per_moment'][label]:.1f}s per moment", flush=True)
    return out


def score(result: dict) -> list[dict]:
    judged = json.loads(JUDGEMENTS_PATH.read_text())
    rows = []
    for label, moments in result["readouts"].items():
        name, tier, k = label.split("/")
        row = {"model": name, "vocab": tier, "k": int(k)}
        for stage in ("before", "after"):
            hits = [w in judged[key] for key, r in moments.items() for w in r[stage]]
            row[f"p@5 {stage}"] = sum(hits) / len(hits)
        # How deep in the embedding's list the reranker reached for its top 5
        row["mean source rank"] = sum(sum(r["moved_from"]) for r in moments.values()) / (5 * len(moments))
        rows.append(row)
    return rows


def review(result: dict) -> None:
    judged = json.loads(JUDGEMENTS_PATH.read_text())
    seen = json.loads((RESULTS_DIR / "eyeball.json").read_text())
    for key in SAMPLE:
        before = {w for m in seen.values() for ws in m[key].values() for w in ws}
        after = {w for moments in result["readouts"].values() for w in moments[key]["after"]}
        new = sorted(after - before - set(judged[key]))
        print(f"{key}: {'; '.join(new)}")


def main() -> None:
    # Every reranked read-out, kept outside data/results so the site export skips it
    path = DATA / "rerank_readouts.json"
    if "--review" in sys.argv:
        review(json.loads(path.read_text()))
        return
    if "--score" not in sys.argv:
        path.write_text(json.dumps(run(), indent=2))
    result = json.loads(path.read_text())
    rows = score(result)
    save("rerank", {"k": result["k"], "seconds_per_moment": result["seconds_per_moment"], "rows": rows,
                    "examples": {label: result["readouts"][label] for label in ("qwen3vl/medium/50", "siglip2/large/50")}})
    print(table(rows, list(rows[0])))


if __name__ == "__main__":
    main()
