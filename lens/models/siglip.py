"""SigLIP 2: an image-text model. A segment is the mean of its frame embeddings.

This is the classic way to get a "video embedding" from an image model, and the baseline the
native video models are compared against.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from lens.models.base import Embedder, device, normalize
from lens.videos import Segment

MODEL_ID = "google/siglip2-base-patch16-256"


def _features(output) -> np.ndarray:
    # transformers 5 returns a model output object; 4.x returned the tensor itself
    tensor = output if isinstance(output, torch.Tensor) else output.pooler_output
    return tensor.float().cpu().numpy()


class SigLIP2FrameAverage(Embedder):
    name = "siglip2"
    dim = 768
    # SigLIP was trained on alt-text; this template is its documented zero-shot prompt
    template = "This is a photo of {}."

    def __init__(self) -> None:
        self.device = device()
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        # SigLIP was trained with max_length padding; other padding degrades the text tower
        inputs = self.processor(
            text=texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        ).to(self.device)
        return _features(self.model.get_text_features(**inputs))

    @torch.no_grad()
    def embed_frames(self, frames: list[Image.Image]) -> np.ndarray:
        """One normalised vector per frame."""
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        return normalize(_features(self.model.get_image_features(**inputs)))

    def _embed_segment(self, segment: Segment) -> np.ndarray:
        frames = [Image.open(p).convert("RGB") for p in segment.frames]
        return self.embed_frames(frames).mean(axis=0)

    def embed_images(self, paths) -> np.ndarray:
        return self.embed_frames([Image.open(p).convert("RGB") for p in paths])
