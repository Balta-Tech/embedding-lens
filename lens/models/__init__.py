"""Video-text embedding models behind one interface.

Every model maps text and video segments into one shared space and returns L2-normalised
float32 vectors, so cosine similarity is a plain dot product.
"""

from lens.models.base import Embedder


def load(name: str) -> Embedder:
    if name == "qwen3vl":
        from lens.models.qwen import Qwen3VLEmbedding

        return Qwen3VLEmbedding()
    if name == "gemma2":
        from lens.models.gemma import EmbeddingGemma2

        return EmbeddingGemma2()
    if name == "siglip2":
        from lens.models.siglip import SigLIP2FrameAverage

        return SigLIP2FrameAverage()
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


MODEL_NAMES = ("qwen3vl", "gemma2", "siglip2")
