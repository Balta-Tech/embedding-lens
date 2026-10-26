"""Qwen3-VL-Embedding-2B: a vision-language model fine-tuned into an embedder.

A segment goes in as a native video (its frames, in order), so the model sees motion and not
just a bag of stills. Uses the reference embedder script that ships with the model weights.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from lens.models.base import Embedder, device
from lens.videos import Segment

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
# The model was trained with an instruction in the system prompt; this is its default
INSTRUCTION = "Represent the user's input."


def _load_reference_embedder(path: Path):
    spec = importlib.util.spec_from_file_location(
        "qwen3_vl_embedding", path / "scripts" / "qwen3_vl_embedding.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["qwen3_vl_embedding"] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


class Qwen3VLEmbedding(Embedder):
    name = "qwen3vl"
    dim = 2048

    def __init__(self, frames_per_segment: int = 10, instruction: str = INSTRUCTION) -> None:
        path = Path(snapshot_download(MODEL_ID, local_files_only=True))
        Qwen3VLEmbedder = _load_reference_embedder(path)
        # num_frames/max_frames = the frames we pass, so the script neither drops nor repeats any
        self.embedder = Qwen3VLEmbedder(
            str(path),
            num_frames=frames_per_segment,
            max_frames=frames_per_segment,
            torch_dtype=torch.bfloat16,
        )
        # The reference script only knows CUDA and CPU
        self.embedder.model.to(device())
        self.instruction = instruction

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = [{"text": t, "instruction": self.instruction} for t in texts]
        return self.embedder.process(inputs).float().cpu().numpy()

    @torch.no_grad()
    def _embed_segment(self, segment: Segment) -> np.ndarray:
        frames = [str(p) for p in segment.frames]
        inputs = [{"video": frames, "instruction": self.instruction}]
        return self.embedder.process(inputs).float().cpu().numpy()[0]
