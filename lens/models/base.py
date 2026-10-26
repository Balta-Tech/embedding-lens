from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
import torch

from lens.videos import Segment


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


class Embedder(ABC):
    name: str
    dim: int
    # How a bare phrase is turned into text for the model, e.g. "a video of {}".
    # The prompt-template experiment overrides this.
    template: str = "{}"

    @abstractmethod
    def _embed_texts(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def _embed_segment(self, segment: Segment) -> np.ndarray: ...

    def embed_texts(
        self,
        phrases: Sequence[str],
        batch_size: int = 64,
        template: str | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        template = template or self.template
        texts = [template.format(p) for p in phrases]
        out = []
        for start in range(0, len(texts), batch_size):
            out.append(self._embed_texts(texts[start : start + batch_size]))
            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
        return normalize(np.concatenate(out))

    def embed_segments(
        self,
        segments: Sequence[Segment],
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        out = []
        for i, segment in enumerate(segments):
            out.append(self._embed_segment(segment))
            if progress:
                progress(i + 1, len(segments))
        return normalize(np.stack(out))
