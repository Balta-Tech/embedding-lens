"""Compute and cache segment and vocabulary embeddings per model.

    uv run python -m lens.index qwen3vl gemma2 siglip2

Vocabulary tiers are nested (small is a prefix of medium, medium of large), so only the large
vocabulary is embedded; smaller tiers are slices of it. Work is saved in chunks, so an
interrupted run resumes where it stopped.
"""

import json
import sys
import time

import numpy as np

from lens import vocab
from lens.config import DATA, EMBEDDINGS_DIR, FRAME_WIDTH
from lens.models import load
from lens.models.base import Embedder
from lens.videos import Segment, all_segments

CHUNK = 4096


def model_dir(model_name: str):
    return EMBEDDINGS_DIR / model_name


def _progress(label: str):
    t0 = time.perf_counter()

    def report(done: int, total: int) -> None:
        rate = done / max(time.perf_counter() - t0, 1e-9)
        print(f"\r{label}: {done}/{total}  {rate:.1f}/s", end="", flush=True)

    return report


def index_segments(model: Embedder, segments: list[Segment]) -> None:
    out = model_dir(model.name)
    if (out / "segments.npy").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    vectors = model.embed_segments(segments, progress=_progress(f"{model.name} segments"))
    print()
    np.save(out / "segments.npy", vectors.astype(np.float16))
    (out / "segments.json").write_text(json.dumps([s.key for s in segments]))


def index_vocab(model: Embedder) -> None:
    phrases = vocab.load("large")
    chunks = model_dir(model.name) / "vocab"
    chunks.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(phrases), CHUNK):
        path = chunks / f"{start // CHUNK:04d}.npy"
        if path.exists():
            continue
        batch = phrases[start : start + CHUNK]
        vectors = model.embed_texts(batch, progress=_progress(f"{model.name} vocab @{start}"))
        print()
        np.save(path, vectors.astype(np.float16))


NULL_KINDS = ("black", "grey", "noise")


def null_segments(frames: int = 10) -> list[Segment]:
    """Content-free videos: all black, all mid-grey, and per-frame random noise."""
    from PIL import Image

    segments = []
    height = FRAME_WIDTH * 9 // 16
    rng = np.random.default_rng(0)
    for kind in NULL_KINDS:
        folder = DATA / "null" / kind
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(frames):
            path = folder / f"{i + 1:05d}.jpg"
            if not path.exists():
                if kind == "noise":
                    pixels = rng.integers(0, 256, (height, FRAME_WIDTH, 3), dtype=np.uint8)
                else:
                    pixels = np.full((height, FRAME_WIDTH, 3), 0 if kind == "black" else 128, np.uint8)
                Image.fromarray(pixels).save(path, quality=90)
            paths.append(path)
        segments.append(Segment(f"null-{kind}", 0, 0.0, frames / 2, tuple(paths)))
    return segments


def index_null(model: Embedder) -> None:
    path = model_dir(model.name) / "null.npy"
    if not path.exists():
        np.save(path, model.embed_segments(null_segments()))


def load_null(model_name: str) -> np.ndarray:
    return np.load(model_dir(model_name) / "null.npy").astype(np.float32)


def load_segments(model_name: str) -> tuple[list[str], np.ndarray]:
    out = model_dir(model_name)
    keys = json.loads((out / "segments.json").read_text())
    return keys, np.load(out / "segments.npy").astype(np.float32)


def load_vocab(model_name: str, tier: str = "large") -> tuple[list[str], np.ndarray]:
    phrases = vocab.load(tier)
    chunks = sorted((model_dir(model_name) / "vocab").glob("*.npy"))
    needed = -(-len(phrases) // CHUNK)
    if len(chunks) < needed:
        raise FileNotFoundError(f"{model_name}: vocabulary '{tier}' not fully indexed yet")
    vectors = np.concatenate([np.load(c) for c in chunks[:needed]])[: len(phrases)]
    return phrases, vectors.astype(np.float32)


def main() -> None:
    segments = all_segments()
    for name in sys.argv[1:]:
        model = load(name)
        index_segments(model, segments)
        index_null(model)
        index_vocab(model)
        del model


if __name__ == "__main__":
    main()
