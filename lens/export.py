"""Export precomputed results for the static site in web/public.

    uv run python -m lens.export

Everything the site shows without the local backend comes from here: read-outs per segment,
concept lanes, preset searches, the offline game table, segment thumbnails and experiment results.
"""

import json
import shutil
import subprocess
from datetime import date

import numpy as np

from lens.config import BEST_METHOD, RESULTS_DIR, ROOT, SEGMENT_SECONDS
from lens.index import load_null, load_segments, load_vocab, model_dir
from lens.models import load
from lens.readout import Space, change, sparse_decompose, unit
from lens.videos import SOURCES, duration_seconds, segments, video_path

OUT = ROOT / "web" / "public"
DATA_OUT = OUT / "data"
MEDIA_OUT = OUT / "media"

MODELS = {
    "qwen3vl": {"label": "Qwen3-VL-Embedding 2B", "kind": "native video", "dim": 2048},
    "gemma2": {"label": "EmbeddingGemma 2", "kind": "native video", "dim": 768},
    "siglip2": {"label": "SigLIP 2", "kind": "frame average", "dim": 768},
}
DEFAULT_MODEL = "qwen3vl"
TIERS = ("small", "medium", "large")
DEFAULT_TIER = "medium"
GAME_TIER = "small"
TOP = 24
SPRITE_FRAMES = 5

LANES = [
    "mountain", "snow", "ice", "rock", "tunnel", "night", "sky", "forest", "sea", "person",
    "face", "hands", "airplane", "car", "road", "crowd", "screen", "text", "video game",
]
QUERIES = [
    "mountains", "snow and ice", "a dark tunnel at night", "a fighter jet over the sea",
    "go-kart racing", "a person's face up close", "a computer screen", "climbing a ladder",
    "a summit cross", "a runway",
]


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))


def _r(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


def _zscore(x: np.ndarray) -> list[float]:
    return [_r(v, 2) for v in (x - x.mean()) / (x.std() + 1e-6)]


def available_models() -> list[str]:
    return [m for m in MODELS if (model_dir(m) / "segments.npy").exists()]


def export_manifest(models: list[str]) -> None:
    videos = []
    for source in SOURCES:
        offset = 0
        if source.section:
            minutes, seconds = source.section.split("-")[0].split(":")
            offset = int(minutes) * 60 + int(seconds)
        videos.append({
            "id": source.id,
            "title": source.title,
            "youtube": source.youtube_id,
            "offset": offset,
            "world": source.world,
            "duration": round(duration_seconds(video_path(source))),
            "segments": len(segments(source)),
        })
    _write(DATA_OUT / "manifest.json", {
        "generated": date.today().isoformat(),
        "segmentSeconds": SEGMENT_SECONDS,
        "spriteFrames": SPRITE_FRAMES,
        "models": [{"id": m, "bestLens": BEST_METHOD[m], **MODELS[m]} for m in models],
        "defaultModel": DEFAULT_MODEL if DEFAULT_MODEL in models else models[0],
        "tiers": [{"id": t, "size": len(load_vocab(models[0], t)[0])} for t in TIERS],
        "defaultTier": DEFAULT_TIER,
        "videos": videos,
    })


def export_model(name: str) -> None:
    keys, videos = load_segments(name)
    null = load_null(name)
    for tier in TIERS:
        phrases, text = load_vocab(name, tier)
        export_readouts(name, tier, Space(phrases, text, videos, null_videos=null), keys, videos)
    phrases, text = load_vocab(name, DEFAULT_TIER)
    space = Space(phrases, text, videos, null_videos=null)

    model = load(name)
    lanes = model.embed_texts(LANES)
    lane_scores = {c: _zscore(_lane(space, videos, lanes[i])) for i, c in enumerate(LANES)}
    _write(DATA_OUT / "lanes" / f"{name}.json", {"concepts": LANES, "scores": lane_scores})

    queries = model.embed_texts(QUERIES)
    _write(DATA_OUT / "search" / f"{name}.json", {
        "queries": QUERIES,
        "scores": {q: _zscore(videos @ queries[i]) for i, q in enumerate(QUERIES)},
    })
    export_game(name, space, keys, videos)
    del model


def export_readouts(name: str, tier: str, space: Space, keys: list[str], videos: np.ndarray):
    readouts = {}
    for i, (key, v) in enumerate(zip(keys, videos)):
        terms, fit = sparse_decompose(v, space, 6)
        readouts[key] = {
            "raw": [[p, _r(s)] for p, s in space.top(v, TOP, "raw")],
            "centered": [[p, _r(s)] for p, s in space.top(v, TOP, "centered")],
            "sparse": {"terms": [[p, _r(w)] for p, w in terms], "fit": _r(fit, 3)},
        }
        # What changed since the previous moment of the same video
        if i > 0 and keys[i - 1].split(":")[0] == key.split(":")[0]:
            amount, appeared, faded = change(space, videos[i - 1], v, 4)
            readouts[key]["change"] = {"amount": _r(amount, 3), "appeared": appeared, "faded": faded}
    _write(DATA_OUT / "readouts" / f"{name}.{tier}.json", {"keys": keys, "segments": readouts})


def _lane(space: Space, videos: np.ndarray, concept: np.ndarray) -> np.ndarray:
    """How strongly each segment shows a concept, in the centred space (gap removed)."""
    vc = unit(videos - space.video_mean)
    return vc @ unit(concept - space.text_mean)


def export_game(name: str, space: Space, keys: list[str], videos: np.ndarray) -> None:
    """Rank of every small-vocabulary word for every segment, so the game works offline."""
    phrases, text = load_vocab(name, GAME_TIER)
    game_space = Space(phrases, text, videos, null_videos=space.null_videos)
    method = BEST_METHOD[name]
    ranks = np.zeros((len(keys), len(phrases)), dtype=np.uint16)
    for i, v in enumerate(videos):
        order = np.argsort(-game_space.scores(v, method))
        ranks[i, order] = np.arange(1, len(phrases) + 1)
    folder = DATA_OUT / "game"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.ranks.bin").write_bytes(ranks.tobytes())
    _write(folder / f"{name}.json", {"keys": keys, "vocab": phrases, "method": method})


def export_media() -> None:
    """One sprite per segment: SPRITE_FRAMES evenly spaced frames side by side, as webp."""
    for source in SOURCES:
        for seg in segments(source):
            out = MEDIA_OUT / source.id / f"{seg.index:03d}.webp"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            picks = np.linspace(0, len(seg.frames) - 1, SPRITE_FRAMES).round().astype(int)
            inputs = []
            for i in picks:
                inputs += ["-i", str(seg.frames[i])]
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
                 "".join(f"[{i}:v]scale=320:-2[f{i}];" for i in range(SPRITE_FRAMES))
                 + "".join(f"[f{i}]" for i in range(SPRITE_FRAMES))
                 + f"hstack=inputs={SPRITE_FRAMES}",
                 "-c:v", "libwebp", "-quality", "62", str(out)],
                check=True,
            )


def export_results() -> None:
    folder = DATA_OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    for path in RESULTS_DIR.glob("*.json"):
        shutil.copy(path, folder / path.name)


def main() -> None:
    models = available_models()
    # Read-outs used to live in one file per model
    for stale in (DATA_OUT / "readouts").glob("*.json"):
        if stale.stem.count(".") == 0:
            stale.unlink()
    export_manifest(models)
    for name in models:
        print(f"exporting {name}")
        export_model(name)
    export_media()
    export_results()


if __name__ == "__main__":
    main()
