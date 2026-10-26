"""Local backend for the live parts of the site: free-text search and scoring typed words.

    uv run uvicorn lens.api:app --port 8000

Models load on first use; only the text side is needed, segment vectors come from the cache.
"""

from functools import cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lens.index import load_segments, load_vocab
from lens.models import MODEL_NAMES, load
from lens.readout import METHODS, Space, softmax

DEFAULT_MODEL = "qwen3vl"
DEFAULT_METHOD = "centered_csls"

app = FastAPI(title="embedding-lens")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://embedding-lens.balta-tech.com", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    # Chrome asks before a public site may call a server on this machine
    allow_private_network=True,
)


@cache
def _model(name: str):
    return load(name)


@cache
def _space(name: str, tier: str) -> tuple[list[str], Space]:
    keys, videos = load_segments(name)
    phrases, text = load_vocab(name, tier)
    return keys, Space(phrases, text, videos, neutral=_model(name).embed_texts(["a video"])[0])


def _check(model: str, method: str) -> None:
    if model not in MODEL_NAMES:
        raise HTTPException(404, f"unknown model {model!r}")
    if method not in METHODS:
        raise HTTPException(404, f"unknown method {method!r}")


def _segment_index(keys: list[str], segment: str) -> int:
    try:
        return keys.index(segment)
    except ValueError:
        raise HTTPException(404, f"unknown segment {segment!r}") from None


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "models": MODEL_NAMES, "default": DEFAULT_MODEL}


@app.get("/api/readout")
def readout(
    segment: str, model: str = DEFAULT_MODEL, method: str = DEFAULT_METHOD,
    tier: str = "large", n: int = 30,
) -> dict:
    _check(model, method)
    keys, space = _space(model, tier)
    v = space.reference[_segment_index(keys, segment)]
    top = space.top(v, n, method)
    weights = softmax(np.array([s for _, s in top]))
    return {"words": [{"phrase": p, "score": s, "weight": float(w)}
                      for (p, s), w in zip(top, weights)]}


class ScoreRequest(BaseModel):
    segment: str
    words: list[str] = Field(min_length=1, max_length=10)
    model: str = DEFAULT_MODEL
    method: str = DEFAULT_METHOD
    tier: str = "large"


@app.post("/api/score")
def score(req: ScoreRequest) -> dict:
    """The game: how close are the player's words to what the model saw?"""
    _check(req.model, req.method)
    keys, space = _space(req.model, req.tier)
    v = space.reference[_segment_index(keys, req.segment)]
    words = [w.strip().lower() for w in req.words if w.strip()][:10]
    if not words:
        raise HTTPException(422, "no words")
    text = _model(req.model).embed_texts(words)
    scores, ranks, percentiles = space.rank_of(v, text, req.method)
    return {
        "vocabulary_size": len(space.phrases),
        "words": [
            {"phrase": w, "score": float(s), "rank": int(r), "percentile": float(p)}
            for w, s, r, p in zip(words, scores, ranks, percentiles)
        ],
        "model_top": [{"phrase": p, "score": s} for p, s in space.top(v, 10, req.method)],
    }


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    model: str = DEFAULT_MODEL


@app.post("/api/search")
def search(req: SearchRequest) -> dict:
    """Text to video: score every segment against a free-text query."""
    _check(req.model, "raw")
    keys, videos = load_segments(req.model)
    q = _model(req.model).embed_texts([req.query])[0]
    sims = videos @ q
    # Relative to the corpus: how unusual is this segment's match for this query?
    z = (sims - sims.mean()) / (sims.std() + 1e-6)
    return {"segments": [{"key": k, "score": float(s), "z": float(zz)}
                         for k, s, zz in zip(keys, sims, z)]}
