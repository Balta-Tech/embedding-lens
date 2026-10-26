"""EmbeddingGemma 2: Google's 740M on-device embedder for text, images, video and audio.

Video goes in as a short clip file, cut from the source video. The model samples frames itself
its default of 1 fps, so it sees half the frames the other models see).
"""

import subprocess

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from lens.config import DATA, FRAME_WIDTH
from lens.models.base import Embedder, device
from lens.videos import SOURCES_BY_ID, Segment, video_path

MODEL_ID = "google/embeddinggemma-2"
CLIPS_DIR = DATA / "clips"


def clip_path(segment: Segment):
    """Cut the segment out of its video as a small mp4, once."""
    out = CLIPS_DIR / segment.video_id / f"{segment.index:04d}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        source = video_path(SOURCES_BY_ID[segment.video_id])
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-ss", f"{segment.start}", "-t",
                f"{segment.end - segment.start}", "-i", str(source),
                "-vf", f"scale={FRAME_WIDTH}:-2", "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-ac", "1", "-ar", "16000", str(out),
            ],
            check=True,
        )
    return out


class EmbeddingGemma2(Embedder):
    name = "gemma2"
    dim = 768
    # Symmetric task prefix: phrases and videos are compared as labels, not as query/document
    template = "task: classification | query: {}"

    def __init__(self) -> None:
        # bfloat16 or float32 only: float16 produces NaNs with this model
        self.model = SentenceTransformer(
            MODEL_ID, device=str(device()), model_kwargs={"torch_dtype": torch.bfloat16}
        )

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, batch_size=len(texts))

    @torch.no_grad()
    def _embed_segment(self, segment: Segment) -> np.ndarray:
        return self.model.encode({"video": str(clip_path(segment))}, convert_to_numpy=True)
