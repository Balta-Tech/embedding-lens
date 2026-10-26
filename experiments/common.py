"""Shared helpers for the mini experiments."""

import json
from functools import cache

import numpy as np

from lens.config import RESULTS_DIR
from lens.index import load_null, load_segments, load_vocab
from lens.models import load
from lens.readout import Space, unit
from lens.videos import SOURCES_BY_ID

NEUTRAL = "a video"


@cache
def model(name: str):
    return load(name)


@cache
def segments(name: str) -> tuple[list[str], np.ndarray]:
    return load_segments(name)


def video_of(key: str) -> str:
    return key.split(":")[0]


def world_of(key: str) -> str:
    return SOURCES_BY_ID[video_of(key)].world


@cache
def space(name: str, tier: str = "large") -> Space:
    phrases, text = load_vocab(name, tier)
    _, videos = segments(name)
    neutral = model(name).embed_texts([NEUTRAL])[0]
    return Space(phrases, text, videos, neutral=neutral, null_videos=load_null(name))


def round_trip(name: str, word_lists: list[list[str]]) -> dict[str, float]:
    """Embed each segment's words as one text and look for that segment among all segments.

    Returns recall@1, recall@5, mean reciprocal rank, and how often the best hit is at least
    from the right video.
    """
    keys, videos = segments(name)
    queries = model(name).embed_texts([", ".join(words) for words in word_lists])
    sims = queries @ videos.T
    order = np.argsort(-sims, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) + 1 for i in range(len(keys))])
    same_video = np.mean([video_of(keys[order[i, 0]]) == video_of(keys[i]) for i in range(len(keys))])
    return {
        "r@1": float(np.mean(ranks == 1)),
        "r@5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1 / ranks)),
        "video@1": float(same_video),
    }


def hubness(word_lists: list[list[str]]) -> dict[str, float]:
    """How much the same few phrases dominate every read-out.

    coverage: distinct phrases / total slots (1 = every segment gets its own words).
    top_share: share of all slots taken by the single most frequent phrase.
    """
    counts: dict[str, int] = {}
    for words in word_lists:
        for w in words:
            counts[w] = counts.get(w, 0) + 1
    slots = sum(counts.values())
    return {"coverage": len(counts) / slots, "top_share": max(counts.values()) / slots}


def second_opinion(judge: str, word_lists: list[list[str]], method: str = "centered") -> float:
    """Mean percentile, under another model, of the words one model chose for each segment.

    100 = the judge also ranks these words above its whole vocabulary for that segment.
    """
    judge_space = space(judge)
    _, judge_videos = segments(judge)
    lookup = _phrase_index(judge)
    out = []
    for v, words in zip(judge_videos, word_lists):
        # Every chosen word comes from the shared vocabulary lists, so the judge already has it
        known = [lookup[w] for w in words if w in lookup]
        text = judge_space.vocab.text[known] if len(known) == len(words) else model(judge).embed_texts(words)
        _, _, pct = judge_space.rank_of(v, text, method)
        out.append(pct.mean())
    return float(np.mean(out))


@cache
def _phrase_index(name: str) -> dict[str, int]:
    return {p: i for i, p in enumerate(space(name).phrases)}


def save(name: str, result) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))


def table(rows: list[dict], columns: list[str]) -> str:
    widths = [max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in columns]
    lines = ["  ".join(c.rjust(w) for c, w in zip(columns, widths))]
    lines += ["  ".join(_fmt(r[c]).rjust(w) for c, w in zip(columns, widths)) for r in rows]
    return "\n".join(lines)


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


__all__ = ["unit"]
