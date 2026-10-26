"""Experiment 3: does a native video embedding know which way time runs?

An image model averaged over frames cannot: reversing or shuffling the frames gives the same
vector. A video model might. Two probes:

1. Reversal: cosine between a segment's embedding and the embedding of the same frames played
   backwards, and with the frames shuffled. 1.0 means time-blind.
2. Take-off vs landing: the F-16 video has a take-off roll and a landing. Played backwards, a
   take-off looks like a landing. For each segment we score "taking off" minus "landing",
   forwards and backwards. A model that reads motion should flip the sign.

    uv run python -m experiments.time_arrow qwen3vl gemma2 siglip2
"""

import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from experiments.common import model, save, table
from lens.config import FRAME_FPS
from lens.readout import unit
from lens.videos import SOURCES_BY_ID, Segment, segments

TAKEOFF = [13, 14, 15]
LANDING = [29, 30, 31, 32]
PROBES = ("a fighter jet taking off from a runway", "a fighter jet landing on a runway")
SAMPLE = 24  # other segments for the reversal probe


def frames_to_clip(frames: tuple[Path, ...], out: Path) -> Path:
    """Gemma takes video files: rebuild a silent clip from an explicit frame order."""
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{f}'\nduration {1 / FRAME_FPS}\n" for f in frames))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-vf", f"fps={FRAME_FPS}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out


def embed(name: str, segment: Segment, frames: tuple[Path, ...], tmp: Path) -> np.ndarray:
    m = model(name)
    if name == "gemma2":
        clip = frames_to_clip(frames, tmp / f"{segment.video_id}-{segment.index}-{hash(frames)}.mp4")
        return unit(m.model.encode({"video": str(clip)}, convert_to_numpy=True))
    variant = Segment(segment.video_id, segment.index, segment.start, segment.end, frames)
    return m.embed_segments([variant])[0]


def run(name: str) -> dict:
    rng = random.Random(0)
    f16 = segments(SOURCES_BY_ID["f16"])
    probed = [f16[i] for i in TAKEOFF + LANDING]
    others = [s for v in SOURCES_BY_ID.values() for s in segments(v) if s not in probed]
    sample = rng.sample(others, SAMPLE)
    probes = model(name).embed_texts(list(PROBES))
    direction = probes[0] - probes[1]
    rev_cos, shuf_cos, flips = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for seg in sample + probed:
            shuffled = list(seg.frames)
            rng.shuffle(shuffled)
            fwd = embed(name, seg, seg.frames, tmp)
            rev = embed(name, seg, seg.frames[::-1], tmp)
            shuf = embed(name, seg, tuple(shuffled), tmp)
            rev_cos.append(float(fwd @ rev))
            shuf_cos.append(float(fwd @ shuf))
            if seg.video_id == "f16" and seg.index in TAKEOFF + LANDING:
                flips.append({
                    "segment": seg.index,
                    "phase": "take-off" if seg.index in TAKEOFF else "landing",
                    "forward": float(fwd @ direction),
                    "backward": float(rev @ direction),
                })
    return {
        "model": name,
        "reversed_cos": float(np.mean(rev_cos)),
        "shuffled_cos": float(np.mean(shuf_cos)),
        "probe": flips,
    }


if __name__ == "__main__":
    results = [run(name) for name in sys.argv[1:]]
    save("time_arrow", results)
    print(table(results, ["model", "reversed_cos", "shuffled_cos"]))
    for r in results:
        print(f"\n{r['model']}: 'taking off' minus 'landing' (positive = reads as take-off)")
        print(table(r["probe"], ["segment", "phase", "forward", "backward"]))
